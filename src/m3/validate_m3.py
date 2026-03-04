"""
M3 验收入口
验收 A: 给定 (p, T) 能稳定返回全部物性
验收 B: 端点/单调性检查通过
"""

import os
import sys
from pathlib import Path

# 脚本位于 src/m3/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir
from physics.pvt import GasPVT
from physics.relperm import RelPermGW
from physics.pvt_validate import PVTValidator
from physics.relperm_validate import RelPermValidator


class M3Validator:
    """M3 物性模块验收器"""
    
    def __init__(self, config_filename: str = 'config.yaml'):
        setup_chinese_support()
        self.logger = setup_logger('M3Validator')
        
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_filename
        self.config = load_config(str(config_path))
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)
        
        self.output_dir = self.config['paths']['outputs']
        ensure_dir(self.output_dir)
        ensure_dir(os.path.join(self.output_dir, 'reports'))
    
    def run(self) -> bool:
        """执行 M3 完整验收"""
        self.logger.info("=" * 60)
        self.logger.info("M3 物性模块验收开始")
        self.logger.info("=" * 60)
        
        all_pass = True
        
        # ============= 验收 A: PVT 物性检查 =============
        self.logger.info("\n>>> 验收 A: PVT 物性（Z, Bg, cg, rho, alphaT）")
        try:
            gas_pvt = GasPVT(config=self.config)
            pvt_validator = PVTValidator(gas_pvt, output_dir=self.output_dir)
            
            # 执行检查
            pvt_pass, pvt_checks = pvt_validator.validate_all()
            for check in pvt_checks:
                self.logger.info(f"  {check}")
            
            if not pvt_pass:
                all_pass = False
                self.logger.error("验收 A: PVT 检查未通过!")
            else:
                self.logger.info("验收 A: PVT 检查全部通过 ✅")
            
            # 出图
            pvt_validator.plot_curves()
            pvt_validator.generate_report()
            
        except Exception as e:
            self.logger.error(f"验收 A 异常: {e}", exc_info=True)
            all_pass = False
        
        # ============= 验收 B: 相渗检查 =============
        self.logger.info("\n>>> 验收 B: 气水相渗（krw, krg 端点/单调/非负）")
        try:
            relperm = RelPermGW(config=self.config)
            rp_validator = RelPermValidator(relperm, output_dir=self.output_dir)
            
            # 执行检查
            rp_pass, rp_checks = rp_validator.validate_all()
            for check in rp_checks:
                self.logger.info(f"  {check}")
            
            if not rp_pass:
                all_pass = False
                self.logger.error("验收 B: 相渗检查未通过!")
            else:
                self.logger.info("验收 B: 相渗检查全部通过 ✅")
            
            # 出图
            rp_validator.plot_curves()
            rp_validator.generate_report()
            
        except Exception as e:
            self.logger.error(f"验收 B 异常: {e}", exc_info=True)
            all_pass = False
        
        # ============= 汇总 =============
        self.logger.info("=" * 60)
        if all_pass:
            self.logger.info("M3 验收结果: ✅ 全部通过")
        else:
            self.logger.info("M3 验收结果: ❌ 存在未通过项")
        self.logger.info("=" * 60)
        
        return all_pass


if __name__ == "__main__":
    validator = M3Validator()
    success = validator.run()
    sys.exit(0 if success else 1)
