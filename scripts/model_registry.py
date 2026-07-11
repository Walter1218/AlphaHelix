"""
模型版本管理模块

功能：
- 自动备份模型文件（带时间戳和版本号）
- 记录模型元数据（特征、超参数、性能指标）
- 支持查询模型历史
- 支持回滚到指定版本
- 支持对比不同版本

用法：
    from model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # 保存模型
    version = registry.save_model(
        model_path="memory/models/gbdt_h10.lightgbm.txt",
        predictions_path="memory/predictions/predictions_h10.parquet",
        config={"features": [...], "params": {...}},
        metrics={"win_rate": 0.638, "cum_excess": 0.4846},
        tags=["best", "production"]
    )
    
    # 查询历史
    history = registry.list_models()
    
    # 回滚
    registry.rollback(version="v3")
"""
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


REGISTRY_DIR = Path("memory/model_registry")
VERSIONS_DIR = REGISTRY_DIR / "versions"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


class ModelRegistry:
    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.versions_dir = registry_dir / "versions"
        self.registry_file = registry_dir / "registry.json"
        self._ensure_dirs()
        self._load_registry()
    
    def _ensure_dirs(self):
        """确保目录存在。"""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self):
        """加载注册表。"""
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "versions": [],
                "current_version": None,
                "created_at": datetime.now().isoformat(),
            }
    
    def _save_registry(self):
        """保存注册表。"""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def _get_next_version(self) -> str:
        """获取下一个版本号。"""
        versions = self.registry.get("versions", [])
        if not versions:
            return "v1"
        last_version = versions[-1]["version"]
        last_num = int(last_version[1:])
        return f"v{last_num + 1}"
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件 MD5 hash。"""
        if not file_path.exists():
            return ""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def save_model(
        self,
        model_path: str,
        predictions_path: str = None,
        config: Dict = None,
        metrics: Dict = None,
        tags: List[str] = None,
        description: str = "",
    ) -> str:
        """
        保存模型版本。
        
        Args:
            model_path: 模型文件路径
            predictions_path: 预测文件路径
            config: 训练配置（特征、超参数等）
            metrics: 性能指标（胜率、累计超额等）
            tags: 标签列表（如 ["best", "production"]）
            description: 版本描述
        
        Returns:
            版本号
        """
        version = self._get_next_version()
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份模型文件
        model_path = Path(model_path)
        model_backup = version_dir / model_path.name
        if model_path.exists():
            shutil.copy2(model_path, model_backup)
        
        # 备份模型元数据
        meta_path = model_path.with_suffix("").with_name(model_path.stem + "_meta.json")
        if meta_path.exists():
            meta_backup = version_dir / meta_path.name
            shutil.copy2(meta_path, meta_backup)
        
        # 备份预测文件
        pred_backup = None
        if predictions_path:
            pred_path = Path(predictions_path)
            if pred_path.exists():
                pred_backup = version_dir / pred_path.name
                shutil.copy2(pred_path, pred_backup)
        
        # 构建版本记录
        version_record = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "tags": tags or [],
            "model_file": model_path.name,
            "model_hash": self._compute_file_hash(model_path),
            "predictions_file": Path(predictions_path).name if predictions_path else None,
            "config": config or {},
            "metrics": metrics or {},
            "files": [f.name for f in version_dir.iterdir()],
        }
        
        # 更新注册表
        self.registry["versions"].append(version_record)
        self.registry["current_version"] = version
        self._save_registry()
        
        print(f"[ModelRegistry] Saved {version}: {model_path.name}")
        print(f"  Metrics: {metrics}")
        print(f"  Tags: {tags}")
        
        return version
    
    def list_models(self, tag: str = None, limit: int = None) -> List[Dict]:
        """
        列出模型版本。
        
        Args:
            tag: 按标签过滤
            limit: 返回数量限制
        
        Returns:
            版本记录列表
        """
        versions = self.registry.get("versions", [])
        
        if tag:
            versions = [v for v in versions if tag in v.get("tags", [])]
        
        if limit:
            versions = versions[-limit:]
        
        return versions
    
    def get_version(self, version: str) -> Optional[Dict]:
        """获取指定版本信息。"""
        for v in self.registry.get("versions", []):
            if v["version"] == version:
                return v
        return None
    
    def get_current_version(self) -> Optional[Dict]:
        """获取当前版本信息。"""
        current = self.registry.get("current_version")
        if current:
            return self.get_version(current)
        return None
    
    def get_best_model(self, metric: str = "win_rate", tag: str = None) -> Optional[Dict]:
        """获取最优模型版本。"""
        versions = self.list_models(tag=tag)
        if not versions:
            return None
        
        # 按指标排序
        valid = [v for v in versions if metric in v.get("metrics", {})]
        if not valid:
            return None
        
        return max(valid, key=lambda v: v["metrics"][metric])
    
    def rollback(self, version: str) -> bool:
        """
        回滚到指定版本。
        
        Args:
            version: 版本号
        
        Returns:
            是否成功
        """
        version_info = self.get_version(version)
        if not version_info:
            print(f"[ModelRegistry] Version {version} not found")
            return False
        
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            print(f"[ModelRegistry] Version directory {version_dir} not found")
            return False
        
        # 恢复模型文件
        model_file = version_info.get("model_file")
        if model_file:
            src = version_dir / model_file
            dst = Path("memory/models") / model_file
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[ModelRegistry] Restored {model_file}")
        
        # 恢复预测文件
        pred_file = version_info.get("predictions_file")
        if pred_file:
            src = version_dir / pred_file
            dst = Path("memory/predictions") / pred_file
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[ModelRegistry] Restored {pred_file}")
        
        # 更新当前版本
        self.registry["current_version"] = version
        self._save_registry()
        
        print(f"[ModelRegistry] Rolled back to {version}")
        return True
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """
        比较两个版本。
        
        Args:
            version1: 版本号1
            version2: 版本号2
        
        Returns:
            比较结果
        """
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        result = {
            "version1": version1,
            "version2": version2,
            "metrics_diff": {},
            "config_diff": {},
        }
        
        # 比较指标
        metrics1 = v1.get("metrics", {})
        metrics2 = v2.get("metrics", {})
        all_metrics = set(list(metrics1.keys()) + list(metrics2.keys()))
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            result["metrics_diff"][metric] = {
                version1: val1,
                version2: val2,
                "diff": val2 - val1,
            }
        
        return result
    
    def print_history(self, limit: int = 10):
        """打印模型历史。"""
        versions = self.list_models(limit=limit)
        
        print(f"\n{'='*80}")
        print(f"Model Registry History (last {limit} versions)")
        print(f"{'='*80}")
        print(f"{'Version':<8} {'Timestamp':<20} {'Win Rate':<10} {'Cum Excess':<12} {'Tags':<20} {'Description'}")
        print(f"{'-'*8} {'-'*20} {'-'*10} {'-'*12} {'-'*20} {'-'*30}")
        
        for v in versions:
            metrics = v.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            cum_excess = metrics.get("cum_excess", 0)
            tags = ", ".join(v.get("tags", []))
            desc = v.get("description", "")[:30]
            timestamp = v.get("timestamp", "")[:19]
            
            print(f"{v['version']:<8} {timestamp:<20} {win_rate:<10.1%} {cum_excess:<12.2%} {tags:<20} {desc}")
        
        current = self.registry.get("current_version")
        print(f"\nCurrent version: {current}")
        print(f"{'='*80}")


def main():
    """CLI 入口。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型版本管理")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出模型版本")
    list_parser.add_argument("--limit", type=int, default=10, help="返回数量限制")
    list_parser.add_argument("--tag", type=str, help="按标签过滤")
    
    # show 命令
    show_parser = subparsers.add_parser("show", help="显示版本详情")
    show_parser.add_argument("version", type=str, help="版本号")
    
    # rollback 命令
    rollback_parser = subparsers.add_parser("rollback", help="回滚到指定版本")
    rollback_parser.add_argument("version", type=str, help="版本号")
    
    # best 命令
    best_parser = subparsers.add_parser("best", help="显示最优模型")
    best_parser.add_argument("--metric", type=str, default="win_rate", help="排序指标")
    best_parser.add_argument("--tag", type=str, help="按标签过滤")
    
    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="比较两个版本")
    compare_parser.add_argument("version1", type=str, help="版本号1")
    compare_parser.add_argument("version2", type=str, help="版本号2")
    
    args = parser.parse_args()
    
    registry = ModelRegistry()
    
    if args.command == "list":
        registry.print_history(limit=args.limit)
    elif args.command == "show":
        v = registry.get_version(args.version)
        if v:
            print(json.dumps(v, ensure_ascii=False, indent=2))
        else:
            print(f"Version {args.version} not found")
    elif args.command == "rollback":
        registry.rollback(args.version)
    elif args.command == "best":
        v = registry.get_best_model(metric=args.metric, tag=args.tag)
        if v:
            print(f"Best model: {v['version']}")
            print(f"  Metrics: {v['metrics']}")
            print(f"  Tags: {v['tags']}")
        else:
            print("No model found")
    elif args.command == "compare":
        result = registry.compare_versions(args.version1, args.version2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        registry.print_history()


if __name__ == "__main__":
    main()
