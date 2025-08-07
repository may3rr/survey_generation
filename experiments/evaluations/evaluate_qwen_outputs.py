import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import json
import pandas as pd
from rouge_score import rouge_scorer
from typing import Dict, List, Optional
import logging
import nltk
nltk.download('punkt')  # 用于分句
class SurveyEvaluator:
    """Survey evaluation class"""
    
    def __init__(self):
        self._setup_logger()
        # 使用与论文相同的ROUGE配置
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.metrics = ['rouge1', 'rouge2', 'rougeL']
        self.score_types = ['precision', 'recall', 'fmeasure']

    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_generated_survey(self, file_path: str) -> Optional[Dict]:
        """加载生成的综述"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading survey file {file_path}: {str(e)}")
            return None

    def postprocess_text(self, text: str) -> str:
        """后处理文本，清理markdown语法和多余的换行符"""
        # 清理文本
        text = text.strip()
        
        # 清除markdown标记
        # 移除以#开头的标题标记
        text = ' '.join(line.lstrip('#').strip() for line in text.split('\n'))
        
        # 清理连续的换行符和空格
        text = ' '.join(text.split())
        
        # 使用简单的规则分句
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return "\n".join(sentences)

    def evaluate_survey(self, generated_survey: Dict, original_df: pd.DataFrame) -> Dict:
        """评估生成的综述 - 使用整体评估方法"""
        try:
            # 找到原始综述
            original_survey = original_df[original_df['title'] == generated_survey['title']]
            if len(original_survey) == 0:
                self.logger.error(f"Original survey not found: {generated_survey['title']}")
                return {}
            original_survey = original_survey.iloc[0]
            
            # 将所有section的文本合并成一个完整文本
            generated_full_text = ""
            original_full_text = ""
            
            # 收集所有sections的文本
            for section_title, generated_content in generated_survey['sections'].items():
                if section_title in original_survey['section']:
                    section_index = original_survey['section'].index(section_title)
                    if section_index < len(original_survey['text']):
                        # 添加分隔符以保持section的边界
                        generated_full_text += generated_content + "\n\n"
                        original_full_text += original_survey['text'][section_index] + "\n\n"
            
            # 文本后处理
            generated_full_text = self.postprocess_text(generated_full_text)
            original_full_text = self.postprocess_text(original_full_text)
            
            # 计算整体ROUGE分数
            try:
                scores = self.scorer.score(original_full_text, generated_full_text)
                return {
                    'overall': {
                        metric: {
                            'precision': scores[metric].precision,
                            'recall': scores[metric].recall,
                            'fmeasure': scores[metric].fmeasure
                        }
                        for metric in self.metrics
                    }
                }
            except Exception as e:
                self.logger.error(f"Error calculating ROUGE scores: {str(e)}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error evaluating survey: {str(e)}")
            return {}

    def calculate_overall_average(self, all_scores: List[Dict]) -> Dict:
        """计算所有文献的总体平均分数"""
        try:
            if not all_scores:
                self.logger.warning("No scores to average")
                return {metric: {score_type: 0.0 for score_type in self.score_types}
                       for metric in self.metrics}
            
            # 初始化累计值
            total_scores = {
                metric: {score_type: 0.0 for score_type in self.score_types}
                for metric in self.metrics
            }
            
            # 累加所有分数
            for doc_scores in all_scores:
                if 'overall' in doc_scores:
                    for metric in self.metrics:
                        for score_type in self.score_types:
                            total_scores[metric][score_type] += doc_scores['overall'][metric][score_type]
            
            # 计算平均值
            num_docs = len(all_scores)
            if num_docs > 0:
                for metric in self.metrics:
                    for score_type in self.score_types:
                        total_scores[metric][score_type] /= num_docs
            
            return total_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating average scores: {str(e)}")
            return {metric: {score_type: 0.0 for score_type in self.score_types}
                   for metric in self.metrics}

    def print_overall_scores(self, scores: Dict):
        """打印总体平均分数"""
        print("\nOverall ROUGE Scores:")
        for metric, metric_scores in scores.items():
            print(f"\n{metric}:")
            for score_type, value in metric_scores.items():
                print(f"  {score_type}: {value:.4f}")

def main():
    try:
        evaluator = SurveyEvaluator()
        project_root = Path(__file__).parent.parent.parent
        
        # 使用配置文件中的路径
        config_path = project_root / 'configs/config.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            raw_data_path = config.get('data', {}).get('raw_data_path', 'data/raw/original_survey_df.pkl')
            output_dir = config.get('output', {}).get('directory', 'qwen3_output')
        else:
            raw_data_path = 'data/raw/original_survey_df.pkl'
            output_dir = 'qwen3_output'
        
        # 加载原始数据
        raw_data_full_path = project_root / raw_data_path
        if not raw_data_full_path.exists():
            print(f"原始数据文件不存在: {raw_data_full_path}")
            print("请先运行数据准备脚本: python prepare_data.py")
            return
            
        original_df = pd.read_pickle(raw_data_full_path)
        
        # 检查输出目录
        output_full_path = project_root / output_dir
        generated_files = list(output_full_path.glob('*generated.json'))
        
        if not generated_files:
            print(f"未找到生成的综述文件在: {output_full_path}")
            print("请先运行实验生成综述")
            return
            
        print(f"找到 {len(generated_files)} 个生成的综述文件")
        
        # 评估所有生成的综述
        all_scores = []
        for input_path in generated_files:
            print(f"正在评估: {input_path.name}")
            generated_survey = evaluator.load_generated_survey(str(input_path))
            if generated_survey:
                scores = evaluator.evaluate_survey(generated_survey, original_df)
                if scores:
                    all_scores.append(scores)
                    # 为每个生成的综述保存单独的评估结果
                    individual_output_path = output_full_path / f"{input_path.stem}_evaluation.json"
                    with open(individual_output_path, 'w') as f:
                        json.dump({
                            'title': generated_survey['title'],
                            'scores': scores
                        }, f, indent=2)
        
        if all_scores:
            # 计算并打印总体平均分数
            overall_average = evaluator.calculate_overall_average(all_scores)
            evaluator.print_overall_scores(overall_average)
        else:
            print("没有成功评估任何综述文件")
        
        # 保存评估结果
        output_path = output_full_path / 'overall_evaluation.json'
        with open(output_path, 'w') as f:
            json.dump({
                'overall_average': overall_average,
                'individual_scores': all_scores
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error in evaluation process: {str(e)}")

if __name__ == "__main__":
    main()