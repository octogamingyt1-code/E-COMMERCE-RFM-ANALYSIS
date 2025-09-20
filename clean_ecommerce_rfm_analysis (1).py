import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class ECommerceRFMAnalysis:
    def __init__(self):
        """Initialize the RFM Analysis class"""
        self.data = None
        self.rfm_data = None
        self.segments = None
        
    def generate_sample_data(self, n_customers=2000):
        """Generate realistic e-commerce transaction data"""
        print("🔄 Generating sample e-commerce dataset...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Customer base
        customer_ids = [f'CUST_{str(i).zfill(5)}' for i in range(1, n_customers + 1)]
        countries = ['United Kingdom', 'France', 'Germany', 'EIRE', 'Spain', 'Netherlands', 'Italy', 'Portugal']
        
        transactions = []
        current_date = datetime(2024, 1, 31)  # Analysis date
        
        for customer_id in customer_ids:
            # Create different customer behavior patterns
            customer_type = np.random.choice(['champion', 'loyal', 'potential', 'new', 'at_risk', 'lost'], 
                                           p=[0.05, 0.15, 0.25, 0.20, 0.20, 0.15])
            
            country = np.random.choice(countries)
            
            # Define behavior patterns based on customer type
            if customer_type == 'champion':
                n_transactions = np.random.randint(15, 35)
                avg_amount = max(50, np.random.normal(150, 30))
                recency_days = np.random.randint(1, 30)
            elif customer_type == 'loyal':
                n_transactions = np.random.randint(8, 20)
                avg_amount = max(40, np.random.normal(120, 25))
                recency_days = np.random.randint(15, 60)
            elif customer_type == 'potential':
                n_transactions = np.random.randint(3, 10)
                avg_amount = max(30, np.random.normal(100, 20))
                recency_days = np.random.randint(30, 90)
            elif customer_type == 'new':
                n_transactions = np.random.randint(1, 4)
                avg_amount = max(25, np.random.normal(80, 15))
                recency_days = np.random.randint(1, 45)
            elif customer_type == 'at_risk':
                n_transactions = np.random.randint(5, 15)
                avg_amount = max(35, np.random.normal(110, 20))
                recency_days = np.random.randint(90, 180)
            else:  # lost
                n_transactions = np.random.randint(1, 8)
                avg_amount = max(30, np.random.normal(90, 15))
                recency_days = np.random.randint(180, 365)
            
            # Generate transactions for this customer
            for _ in range(n_transactions):
                # Random transaction date
                days_ago = np.random.randint(recency_days, min(recency_days + 300, 365))
                transaction_date = current_date - timedelta(days=days_ago)
                
                # Transaction amount with controlled variance (ensure positive)
                std_dev = avg_amount * 0.2  # Reduced variance to 20%
                amount = max(10, np.random.normal(avg_amount, std_dev))
                
                transactions.append({
                    'CustomerID': customer_id,
                    'InvoiceDate': transaction_date,
                    'Amount': round(amount, 2),
                    'Country': country,
                    'CustomerType': customer_type  # For validation
                })
        
        self.data = pd.DataFrame(transactions)
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
        
        print(f"✅ Generated {len(self.data)} transactions for {len(customer_ids)} customers")
        return self.data
    
    def calculate_rfm_scores(self, analysis_date='2024-01-31'):
        """Calculate RFM scores for each customer"""
        print("📊 Calculating RFM scores...")
        
        analysis_date = pd.to_datetime(analysis_date)
        
        # Calculate RFM metrics
        rfm = self.data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
            'Amount': ['count', 'sum']  # Frequency & Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm.reset_index(inplace=True)
        
        # Calculate RFM scores using quintiles
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to int
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Create RFM segment
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        self.rfm_data = rfm
        print("✅ RFM scores calculated successfully")
        return rfm
    
    def segment_customers(self):
        """Segment customers based on RFM scores"""
        print("🎯 Segmenting customers...")
        
        def segment_customers_func(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                return 'Champions'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Loyal Customers'
            elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                return 'New Customers'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
                return 'Potential Loyalists'
            elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
                return 'At Risk'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 3:
                return "Can't Lose Them"
            elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                return 'Hibernating'
            else:
                return 'Lost Customers'
        
        self.rfm_data['Segment'] = self.rfm_data.apply(segment_customers_func, axis=1)
        
        # Calculate segment statistics
        segment_stats = self.rfm_data.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
        segment_stats['Percentage'] = (segment_stats['Customer_Count'] / len(self.rfm_data) * 100).round(1)
        segment_stats = segment_stats.sort_values('Customer_Count', ascending=False)
        
        self.segments = segment_stats
        print("✅ Customer segmentation completed")
        return segment_stats
    
    def create_clean_visualizations(self):
        """Create clean, professional visualizations"""
        print("📈 Creating professional visualizations...")
        
        # Define color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#90A959', '#F2CC8F']
        
        # 1. Customer Segments Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('E-Commerce Customer Segmentation Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # Pie chart - Customer distribution
        segment_counts = self.rfm_data['Segment'].value_counts()
        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Customer Distribution by Segment', fontsize=14, fontweight='bold', pad=20)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Bar chart - Revenue by segment
        revenue_by_segment = self.rfm_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
        bars = ax2.barh(revenue_by_segment.index, revenue_by_segment.values, color=colors[:len(revenue_by_segment)])
        ax2.set_title('Total Revenue by Segment', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Revenue ($)', fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        # Customer count by segment
        segment_counts_sorted = segment_counts.sort_values(ascending=True)
        bars2 = ax3.barh(segment_counts_sorted.index, segment_counts_sorted.values, color=colors[:len(segment_counts_sorted)])
        ax3.set_title('Customer Count by Segment', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Number of Customers', fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # Average Order Value by segment
        aov_by_segment = (self.rfm_data.groupby('Segment')['Monetary'].sum() / 
                         self.rfm_data.groupby('Segment')['Frequency'].sum()).sort_values(ascending=True)
        bars3 = ax4.barh(aov_by_segment.index, aov_by_segment.values, color=colors[:len(aov_by_segment)])
        ax4.set_title('Average Order Value by Segment', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('AOV ($)', fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars3):
            width = bar.get_width()
            ax4.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'${width:.0f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 2. RFM Analysis Deep Dive
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RFM Metrics Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # RFM Distribution - Create individual histograms
        ax1.hist(self.rfm_data['Recency'], bins=25, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_title('Recency Distribution (Days)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Days Since Last Purchase')
        ax1.set_ylabel('Number of Customers')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(self.rfm_data['Frequency'], bins=25, color='#A23B72', alpha=0.7, edgecolor='black')
        ax2.set_title('Frequency Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Purchases')
        ax2.set_ylabel('Number of Customers')
        ax2.grid(True, alpha=0.3)
        
        ax3.hist(self.rfm_data['Monetary'], bins=25, color='#F18F01', alpha=0.7, edgecolor='black')
        ax3.set_title('Monetary Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Total Amount Spent ($)')
        ax3.set_ylabel('Number of Customers')
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: Recency vs Frequency colored by Monetary
        scatter = ax4.scatter(self.rfm_data['Recency'], self.rfm_data['Frequency'], 
                            c=self.rfm_data['Monetary'], cmap='viridis', alpha=0.6, s=50)
        ax4.set_xlabel('Recency (Days)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Customer Distribution (Recency vs Frequency)', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Monetary Value ($)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Segment Performance Heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # RFM scores by segment
        rfm_summary = self.rfm_data.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean()
        
        # Create heatmap
        sns.heatmap(rfm_summary.T, annot=True, cmap='RdYlBu_r', center=3, 
                   fmt='.1f', cbar_kws={'shrink': .8}, ax=ax)
        ax.set_title('RFM Scores by Customer Segment', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Customer Segments', fontweight='bold')
        ax.set_ylabel('RFM Metrics', fontweight='bold')
        
        # Customize labels
        ax.set_yticklabels(['Recency Score', 'Frequency Score', 'Monetary Score'])
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Clean visualizations created successfully")
    
    def display_key_metrics(self):
        """Display key business metrics in a clean format"""
        print("\n" + "="*70)
        print("📊 KEY BUSINESS METRICS DASHBOARD")
        print("="*70)
        
        total_customers = len(self.rfm_data)
        total_revenue = self.rfm_data['Monetary'].sum()
        avg_order_value = total_revenue / self.rfm_data['Frequency'].sum()
        avg_customer_value = total_revenue / total_customers
        
        print(f"""
🔢 CUSTOMER METRICS:
   • Total Customers: {total_customers:,}
   • Average Customer Value: ${avg_customer_value:,.2f}
   • Average Purchase Frequency: {self.rfm_data['Frequency'].mean():.1f} orders per customer
   
💰 REVENUE METRICS:
   • Total Revenue: ${total_revenue:,.2f}
   • Average Order Value: ${avg_order_value:.2f}
   • Revenue per Customer: ${total_revenue/total_customers:,.2f}
   
📅 ENGAGEMENT METRICS:
   • Average Days Since Last Purchase: {self.rfm_data['Recency'].mean():.0f} days
   • Most Recent Purchase: {self.rfm_data['Recency'].min():.0f} days ago
   • Oldest Customer Activity: {self.rfm_data['Recency'].max():.0f} days ago
        """)
        
        print("="*70)
    
    def display_segment_summary(self):
        """Display segment summary in a clean table format"""
        print("\n📋 CUSTOMER SEGMENT SUMMARY")
        print("-"*80)
        
        # Create a formatted summary
        summary_data = []
        for segment in self.segments.index:
            count = self.segments.loc[segment, 'Customer_Count']
            percentage = self.segments.loc[segment, 'Percentage']
            avg_monetary = self.segments.loc[segment, 'Avg_Monetary']
            avg_frequency = self.segments.loc[segment, 'Avg_Frequency']
            avg_recency = self.segments.loc[segment, 'Avg_Recency']
            
            total_revenue = count * avg_monetary
            
            summary_data.append({
                'Segment': segment,
                'Customers': f"{count} ({percentage}%)",
                'Avg Value': f"${avg_monetary:,.0f}",
                'Avg Orders': f"{avg_frequency:.1f}",
                'Avg Recency': f"{avg_recency:.0f} days",
                'Total Revenue': f"${total_revenue:,.0f}"
            })
        
        # Display as formatted table
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        print("-"*80)
    
    def generate_marketing_recommendations(self):
        """Generate focused marketing recommendations"""
        print("\n💡 STRATEGIC MARKETING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {
            'Champions': {
                'priority': '🏆 HIGHEST',
                'strategy': 'VIP Retention',
                'key_actions': [
                    'Exclusive early access to new products',
                    'Dedicated premium customer service',
                    'Referral rewards program',
                    'Personalized thank-you campaigns'
                ],
                'budget': '$50-100 per customer',
                'expected_roi': '300-500%'
            },
            'At Risk': {
                'priority': '🚨 URGENT',
                'strategy': 'Win-Back Campaign',
                'key_actions': [
                    'Limited-time 25% discount offers',
                    '"We miss you" personalized emails',
                    'Free shipping incentives',
                    'Product recommendations based on history'
                ],
                'budget': '$25-40 per customer',
                'expected_roi': '150-250%'
            },
            'Potential Loyalists': {
                'priority': '🌱 HIGH OPPORTUNITY',
                'strategy': 'Relationship Building',
                'key_actions': [
                    'Personalized product recommendations',
                    'Member-only exclusive discounts',
                    'Educational content and tutorials',
                    'Social media engagement'
                ],
                'budget': '$20-30 per customer',
                'expected_roi': '200-300%'
            }
        }
        
        # Display top 3 priority segments
        priority_segments = ['Champions', 'At Risk', 'Potential Loyalists']
        
        for segment in priority_segments:
            if segment in self.segments.index and segment in recommendations:
                rec = recommendations[segment]
                customer_count = self.segments.loc[segment, 'Customer_Count']
                
                print(f"\n{rec['priority']} - {segment.upper()} ({customer_count} customers)")
                print(f"Strategy: {rec['strategy']}")
                print("Key Actions:")
                for action in rec['key_actions']:
                    print(f"  • {action}")
                print(f"Budget: {rec['budget']}")
                print(f"Expected ROI: {rec['expected_roi']}")
                print("-" * 40)

def main():
    """Main execution function with clean output"""
    print("🎯 E-COMMERCE RFM ANALYSIS")
    print("=" * 40)
    
    # Initialize analysis
    analyzer = ECommerceRFMAnalysis()
    
    # Execute analysis pipeline
    analyzer.generate_sample_data(2000)
    analyzer.calculate_rfm_scores()
    analyzer.segment_customers()
    
    # Display results
    analyzer.display_key_metrics()
    analyzer.display_segment_summary()
    
    # Create visualizations
    analyzer.create_clean_visualizations()
    
    # Marketing recommendations
    analyzer.generate_marketing_recommendations()
    
    print("\n✅ ANALYSIS COMPLETED!")
    print("📊 Review the visualizations and implement the marketing strategies above.")
    
    return analyzer

# Execute the analysis
if __name__ == "__main__":
    analyzer = main()