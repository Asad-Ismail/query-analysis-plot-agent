"""Visualization Agent with Structured Outputs"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from typing import Dict, Optional,Set
import logging
from src.models import ChartRecommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_CHART_TYPES: Set[str] = {"bar", "line", "pie", "scatter", "table"}

class VisualizationAgent:
    """Agent for creating styled visualizations with structured chart recommendations"""
    
    def __init__(self, llm: ChatOpenAI, style_config: dict):
        """Initialize the visualization agent
        
        Args:
            llm: ChatOpenAI instance for LLM calls
            style_config: Dict with styling configuration (colors, figure_size, dpi)
        """
        self.llm = llm
        self.style = style_config
        self._apply_style()
        logger.info("VisualizationAgent initialized")
    
    def _apply_style(self):
        """Apply company styling to matplotlib/seaborn"""
        if 'colors' in self.style:
            sns.set_palette(self.style['colors'])
        
        plt.rcParams['figure.figsize'] = self.style.get('figure_size', (10, 6))
        plt.rcParams['figure.dpi'] = self.style.get('dpi', 100)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'
    
    def visualize(
        self, 
        data: pd.DataFrame, 
        query: str, 
        output_dir: str = "outputs",
        chart_type_override: Optional[str] = None
    ) -> Dict:
        """Create visualization based on data and query
        
        Args:
            data: DataFrame to visualize
            query: Original user query for context
            output_dir: Directory to save visualization
            
        Returns:
            Dict with status, filepath, chart_type, and metadata
        """
        try:
            logger.info(f"Starting visualization for query: {query}")
            
            # Validate data
            if data.empty:
                logger.warning("Empty dataset provided")
                return {
                    "status": "error",
                    "error": "Cannot visualize empty dataset"
                }
            
            if len(data.columns) < 2:
                logger.warning("Insufficient columns for visualization")
                return {
                    "status": "error",
                    "error": "Need at least 2 columns for visualization"
                }
            
            if chart_type_override and chart_type_override in VALID_CHART_TYPES:
                logger.info(f"User override '{chart_type_override}' detected. Getting AI column suggestions first...")
                # Get AI recommendation *only for its column suggestions*
                ai_recommendation = self._get_chart_recommendation(data, query)
                
                logger.info(f"Applying user chart type '{chart_type_override}' with AI-suggested columns.")
                recommendation = ChartRecommendation(
                    chart_type=chart_type_override,
                    reasoning=f"User selected '{chart_type_override}' chart type.",
                    x_column=ai_recommendation.x_column,  
                    y_column=ai_recommendation.y_column  
                )

            else:
                if chart_type_override:
                    # Log a warning if the override was invalid (e.g., "auto"), then fall back to AI
                    logger.warning(f"Invalid or unhandled chart_type_override '{chart_type_override}'. Falling back to AI.")
                
                # Default behavior: Get the AI's full recommendation
                logger.info("Getting LLM chart recommendation...")
                recommendation = self._get_chart_recommendation(data, query)

            # Create the chart
            fig, ax = plt.subplots()
            
            if recommendation.chart_type == 'bar':
                self._create_bar_chart(data, ax, recommendation)
            elif recommendation.chart_type == 'line':
                self._create_line_chart(data, ax, recommendation)
            elif recommendation.chart_type == 'pie':
                self._create_pie_chart(data, ax, recommendation)
            elif recommendation.chart_type == 'scatter':
                self._create_scatter_chart(data, ax, recommendation)
            else:  # table or fallback
                self._create_bar_chart(data, ax, recommendation)  # Default to bar
            
            # Apply title and branding
            title = query[:60] + "..." if len(query) > 60 else query
            ax.set_title(title, pad=20)
            
            # Add company branding watermark
            '''
            ax.text(
                0.99, 0.01, 
                'Company Analytics ©',
                transform=ax.transAxes, 
                fontsize=8, 
                alpha=0.5,
                ha='right',
                va='bottom'
            )
            '''
            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"viz_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {filepath}")
            
            return {
                "status": "success",
                "filepath": filepath,
                "chart_type": recommendation.chart_type,
                "reasoning": recommendation.reasoning
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return {
                "status": "error",
                "error": f"Visualization failed: {str(e)}"
            }
    
    def _get_chart_recommendation(
        self, 
        df: pd.DataFrame, 
        query: str
    ) -> ChartRecommendation:
        """Get chart type recommendation using LLM with structured output
        
        Args:
            df: DataFrame to visualize
            query: User's original query
            
        Returns:
            ChartRecommendation with chart type and column suggestions
        """
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(df)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Recommend the best chart type.

            Available chart types:
            - bar: Compare categories or show rankings
            - line: Show trends over time or continuous data
            - pie: Show proportions (max 8 categories)
            - scatter: Show relationships between two numeric variables
            - table: When data is better shown in tabular format

            Consider:
            1. Data types (categorical vs numeric)
            2. Number of categories (pie charts only for <=8)
            3. Presence of time-series data
            4. Query context"""),
            ("user", """Query: {query}

            Data Characteristics:
            {characteristics}

            Column Names: {columns}
            Row Count: {row_count}

            Recommend the best chart type and specify which columns to use.""")])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(ChartRecommendation)
        
        result = structured_llm.invoke(
            prompt.format_messages(
                query=query,
                characteristics=data_characteristics,
                columns=", ".join(df.columns.tolist()),
                row_count=len(df)
            )
        )
        
        # Validate column recommendations
        if result.x_column and result.x_column not in df.columns:
            result.x_column = df.columns[0]
        if result.y_column and result.y_column not in df.columns:
            result.y_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        return result
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> str:
        """Analyze data to help with chart selection"""
        characteristics = []
        
        # Column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        characteristics.append(f"Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:3])})")
        characteristics.append(f"Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols[:3])})")
        
        if datetime_cols:
            characteristics.append(f"DateTime columns: {', '.join(datetime_cols)}")
        
        # Unique values in first column (for pie chart consideration)
        if len(categorical_cols) > 0:
            unique_count = df[categorical_cols[0]].nunique()
            characteristics.append(f"Unique categories in '{categorical_cols[0]}': {unique_count}")
        
        return "\n".join(characteristics)
    
    def _create_bar_chart(
        self, 
        df: pd.DataFrame, 
        ax, 
        recommendation: ChartRecommendation
    ):
        """Create bar chart"""
        x_col = recommendation.x_column or df.columns[0]
        y_col = recommendation.y_column or df.columns[1]
        
        df_plot = df
        
        ax.bar(range(len(df_plot)), df_plot[y_col], color=self.style['colors'][0])
        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot[x_col], rotation=45, ha='right')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(axis='y', alpha=0.3)
    
    def _create_line_chart(
        self, 
        df: pd.DataFrame, 
        ax, 
        recommendation: ChartRecommendation
    ):
        """Create line chart"""
        x_col = recommendation.x_column or df.columns[0]
        y_col = recommendation.y_column or df.columns[1]
        
        ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-labels if they're long
        if df[x_col].dtype == 'object':
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _create_pie_chart(
        self, 
        df: pd.DataFrame, 
        ax, 
        recommendation: ChartRecommendation
    ):
        """Create pie chart"""
        labels_col = recommendation.x_column or df.columns[0]
        values_col = recommendation.y_column or df.columns[1]
        
        # Limit to top 8 categories
        df_plot = df.nlargest(8, values_col) if pd.api.types.is_numeric_dtype(df[values_col]) else df.head(8)
        df_plot = df 
        
        colors = self.style['colors'][:len(df_plot)]
        ax.pie(
            df_plot[values_col], 
            labels=df_plot[labels_col], 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.axis('equal')
    
    def _create_scatter_chart(
        self, 
        df: pd.DataFrame, 
        ax, 
        recommendation: ChartRecommendation
    ):
        """Create scatter plot"""
        x_col = recommendation.x_column or df.columns[0]
        y_col = recommendation.y_column or df.columns[1]
        
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color=self.style['colors'][0])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)