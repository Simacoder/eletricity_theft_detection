import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

class DataAnalyzer:
    def __init__(self):
        """Initialize the DataAnalyzer"""
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Analyze temporal patterns in the data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, go.Figure]: Dictionary of plotly figures
        """
        plots = {}
        
        try:
            # Hourly patterns
            hourly_demand = df.groupby('Hour')['Residual_Demand'].agg(['mean', 'std']).reset_index()
            plots['hourly_pattern'] = px.line(
                hourly_demand,
                x='Hour',
                y='mean',
                error_y='std',
                title='Hourly Demand Pattern with Standard Deviation',
                labels={'mean': 'Average Demand', 'Hour': 'Hour of Day'}
            )
            
            # Weekly patterns
            weekly_demand = df.groupby('DayOfWeek')['Residual_Demand'].mean().reset_index()
            plots['weekly_pattern'] = px.bar(
                weekly_demand,
                x='DayOfWeek',
                y='Residual_Demand',
                title='Weekly Demand Pattern',
                labels={'Residual_Demand': 'Average Demand', 'DayOfWeek': 'Day of Week'}
            )
            
            # Monthly patterns
            monthly_demand = df.groupby('Month')['Residual_Demand'].agg(['mean', 'std']).reset_index()
            plots['monthly_pattern'] = px.line(
                monthly_demand,
                x='Month',
                y='mean',
                error_y='std',
                title='Monthly Demand Pattern',
                labels={'mean': 'Average Demand', 'Month': 'Month'}
            )
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error in temporal pattern analysis: {str(e)}")
            raise
            
    def analyze_system_performance(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Analyze system performance metrics
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, go.Figure]: Dictionary of plotly figures
        """
        plots = {}
        
        try:
            # System efficiency over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date Time Hour Beginning'],
                y=df['System_Efficiency'],
                mode='lines',
                name='System Efficiency'
            ))
            fig.update_layout(title='System Efficiency Over Time')
            plots['efficiency_trend'] = fig
            
            # Generation vs Demand scatter
            plots['generation_demand'] = px.scatter(
                df,
                x='Residual_Demand',
                y='Dispatchable Generation',
                color='System_Efficiency',
                title='Generation vs Demand Relationship',
                labels={
                    'Residual_Demand': 'Residual Demand',
                    'Dispatchable Generation': 'Dispatchable Generation',
                    'System_Efficiency': 'System Efficiency'
                }
            )
            
            # Loss factors analysis
            loss_factors = df[['Total PCLF', 'Total UCLF', 'Total OCLF']].mean()
            plots['loss_factors'] = px.pie(
                values=loss_factors.values,
                names=loss_factors.index,
                title='Distribution of Loss Factors'
            )
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error in system performance analysis: {str(e)}")
            raise
            
    def analyze_renewable_energy(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Analyze renewable energy patterns
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, go.Figure]: Dictionary of plotly figures
        """
        plots = {}
        
        try:
            # RE mix over time
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=df['Date Time Hour Beginning'], y=df['Total RE'],
                          name="Total RE Generation"),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date Time Hour Beginning'], y=df['RE_Mix'],
                          name="RE Mix Ratio"),
                secondary_y=True,
            )
            
            fig.update_layout(title='Renewable Energy Generation and Mix Over Time')
            plots['re_trends'] = fig
            
            # RE utilization by type
            re_types = ['Wind', 'PV', 'CSP', 'Other RE']
            re_capacity = ['Wind Installed Capacity', 'PV Installed Capacity', 
                         'CSP Installed Capacity', 'Other RE Installed Capacity']
            
            utilization_rates = []
            for re, cap in zip(re_types, re_capacity):
                util = (df[re] / df[cap].replace(0, np.nan)).mean()
                utilization_rates.append({'Type': re, 'Utilization': util})
            
            plots['re_utilization'] = px.bar(
                pd.DataFrame(utilization_rates),
                x='Type',
                y='Utilization',
                title='Renewable Energy Utilization Rates',
                labels={'Utilization': 'Average Utilization Rate'}
            )
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error in renewable energy analysis: {str(e)}")
            raise
            
    def generate_feature_correlations(self, df: pd.DataFrame) -> go.Figure:
        """
        Generate correlation heatmap for features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            go.Figure: Correlation heatmap
        """
        try:
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title='Feature Correlations',
                aspect='auto'
            )
            
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {str(e)}")
            raise
            
    def analyze_statistical_distributions(self, df: pd.DataFrame, 
                                       columns: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Analyze statistical distributions of features
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (Optional[List[str]]): Specific columns to analyze
            
        Returns:
            Dict[str, go.Figure]: Dictionary of distribution plots
        """
        plots = {}
        
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
                
            for col in columns:
                # Create distribution plot
                fig = go.Figure()
                
                # Add histogram
                fig.add_trace(go.Histogram(
                    x=df[col],
                    name='Distribution',
                    nbinsx=50,
                    opacity=0.7
                ))
                
                # Add KDE
                kde_x = np.linspace(df[col].min(), df[col].max(), 100)
                kde = df[col].plot.kde()
                kde_y = kde.get_figure().gca().get_lines()[0].get_ydata()
                
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y * len(df[col]) * (df[col].max() - df[col].min()) / 50,
                    name='KDE',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title=f'Distribution of {col}',
                    xaxis_title=col,
                    yaxis_title='Count',
                    showlegend=True
                )
                
                plots[col] = fig
                
            return plots
            
        except Exception as e:
            self.logger.error(f"Error analyzing statistical distributions: {str(e)}")
            raise
            
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive summary report of the data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Summary statistics and insights
        """
        try:
            summary = {
                'basic_stats': {},
                'temporal_patterns': {},
                'system_metrics': {},
                'data_quality': {}
            }
            
            # Basic statistics
            summary['basic_stats'] = {
                'total_records': len(df),
                'date_range': {
                    'start': df['Date Time Hour Beginning'].min(),
                    'end': df['Date Time Hour Beginning'].max()
                },
                'key_metrics': {
                    col: {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    } for col in ['Residual_Demand', 'System_Efficiency', 'Total RE']
                }
            }
            
            # Temporal patterns
            summary['temporal_patterns'] = {
                'hourly_peak': df.groupby('Hour')['Residual_Demand'].mean().idxmax(),
                'weekly_pattern': df.groupby('DayOfWeek')['Residual_Demand'].mean().to_dict(),
                'monthly_pattern': df.groupby('Month')['Residual_Demand'].mean().to_dict()
            }
            
            # System metrics
            summary['system_metrics'] = {
                'avg_system_efficiency': df['System_Efficiency'].mean(),
                'avg_re_contribution': (df['Total RE'] / df['Dispatchable Generation']).mean(),
                'loss_factors': {
                    'PCLF': df['Total PCLF'].mean(),
                    'UCLF': df['Total UCLF'].mean(),
                    'OCLF': df['Total OCLF'].mean()
                }
            }
            
            # Data quality
            summary['data_quality'] = {
                'missing_values': df.isnull().sum().to_dict(),
                'completeness_ratio': 1 - (df.isnull().sum() / len(df)).mean()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise