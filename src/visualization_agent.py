"""Visualization Agent"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

class VisualizationAgent:
    """Agent for creating styled visualizations"""
    
    def __init__(self, llm: ChatOpenAI, style_config: dict):
        self.llm = llm
        self.style = style_config
        self._apply_style()
    
    def _apply_style(self):
        """Apply company styling"""
        sns.set_palette(self.style.get('colors', 'Set2'))
        plt.rcParams['figure.figsize'] = self.style.get('figure_size', (10, 6))
        plt.rcParams['figure.dpi'] = self.style.get('dpi', 100)
    
    def visualize(self, data: pd.DataFrame, query: str, output_dir: str = "outputs") -> dict:
        """Create visualization"""
        chart_type = self._determine_chart_type(data, query)
        
        fig, ax = plt.subplots()
        
        # Create chart based on type
        if chart_type == 'bar':
            self._bar_chart(data, ax)
        elif chart_type == 'line':
            self._line_chart(data, ax)
        elif chart_type == 'pie':
            self._pie_chart(data, ax)
        else:
            self._bar_chart(data, ax)
        
        # Apply branding
        ax.set_title(query[:60], fontweight='bold')
        ax.text(0.99, 0.01, 'Company Analytics ©',
                transform=ax.transAxes, fontsize=8, alpha=0.5,
                ha='right')
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.style['dpi'])
        plt.close()
        
        return {"status": "success", "filepath": filepath, "chart_type": chart_type}
    
    def _determine_chart_type(self, df: pd.DataFrame, query: str) -> str:
        """Determine chart type using LLM"""
        prompt = ChatPromptTemplate.from_template(
            "Given data with columns {cols} and query '{query}', suggest: bar, line, or pie"
        )
        response = self.llm.invoke(prompt.format(cols=list(df.columns), query=query))
        chart_type = response.content.strip().lower()
        return chart_type if chart_type in ['bar', 'line', 'pie'] else 'bar'
    
    def _bar_chart(self, df, ax):
        x, y = df.columns[0], df.columns[1]
        df_plot = df.head(10)
        ax.bar(range(len(df_plot)), df_plot[y])
        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot[x], rotation=45, ha='right')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    
    def _line_chart(self, df, ax):
        x, y = df.columns[0], df.columns[1]
        ax.plot(df[x], df[y], marker='o')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
    
    def _pie_chart(self, df, ax):
        labels, values = df.columns[0], df.columns[1]
        df_plot = df.head(8)
        ax.pie(df_plot[values], labels=df_plot[labels], autopct='%1.1f%%')