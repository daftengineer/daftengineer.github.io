---
title: Thompson Sampling in Production - Building Adaptive Recommendation Systems with Multi-Armed Bandits
tags: machine-learning multi-armed-bandits thompson-sampling recommendation-systems aws production
article_header:
  type: overlay
  theme: dark
  background_color: '#059669'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(5, 150, 105, .4), rgba(168, 85, 247, .4))'
---

Traditional A/B testing locks you into static experiments for weeks or months. But what if your recommendation system could learn and adapt in real-time, automatically shifting traffic to better-performing variants? Enter multi-armed bandits with Thompson sampling—a principled approach to balancing exploration and exploitation that transformed our recommendation engine's performance.

<!--more-->

## The Problem with Static A/B Tests

Our e-commerce client was running traditional A/B tests for their product recommendation algorithm. They had three variants:
- **Collaborative Filtering**: User-based recommendations
- **Content-Based**: Item feature similarity  
- **Hybrid Approach**: Combination of both methods

The challenge? Each test took 6-8 weeks to reach statistical significance, during which 50% of users were stuck with potentially inferior recommendations. Meanwhile, business conditions changed, seasonal patterns emerged, and user preferences evolved—but their testing framework couldn't adapt.

## Multi-Armed Bandits: Adaptive Experimentation

Multi-armed bandits (MAB) solve this problem by treating each recommendation algorithm as an "arm" of a slot machine. Instead of fixed traffic allocation, the system learns which arms perform better and gradually shifts more traffic to them while still exploring alternatives.

### Thompson Sampling Algorithm

Thompson sampling is a Bayesian approach that maintains probability distributions for each arm's performance and samples from these distributions to make decisions:

```python
import numpy as np
from scipy import stats
import boto3
from typing import Dict, List, Tuple

class ThompsonSamplingMAB:
    def __init__(self, arms: List[str], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson Sampling Multi-Armed Bandit
        
        Args:
            arms: List of algorithm names
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.arms = arms
        self.n_arms = len(arms)
        
        # Beta distribution parameters for each arm
        self.alpha = {arm: alpha_prior for arm in arms}
        self.beta = {arm: beta_prior for arm in arms}
        
        # Tracking metrics
        self.total_pulls = {arm: 0 for arm in arms}
        self.total_rewards = {arm: 0 for arm in arms}
        
    def select_arm(self) -> str:
        """Select arm using Thompson sampling"""
        samples = {}
        
        for arm in self.arms:
            # Sample from Beta distribution
            samples[arm] = np.random.beta(self.alpha[arm], self.beta[arm])
            
        # Select arm with highest sample
        selected_arm = max(samples.keys(), key=lambda k: samples[k])
        return selected_arm
        
    def update(self, arm: str, reward: float):
        """Update arm statistics with observed reward"""
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward
        
        # Update Beta parameters
        if reward > 0:  # Success
            self.alpha[arm] += 1
        else:  # Failure
            self.beta[arm] += 1
            
    def get_arm_statistics(self) -> Dict:
        """Get current statistics for all arms"""
        stats = {}
        for arm in self.arms:
            if self.total_pulls[arm] > 0:
                empirical_mean = self.total_rewards[arm] / self.total_pulls[arm]
                # Beta distribution mean
                theoretical_mean = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
                # Confidence interval
                ci_lower = stats.beta.ppf(0.025, self.alpha[arm], self.beta[arm])
                ci_upper = stats.beta.ppf(0.975, self.alpha[arm], self.beta[arm])
            else:
                empirical_mean = theoretical_mean = ci_lower = ci_upper = 0
                
            stats[arm] = {
                'pulls': self.total_pulls[arm],
                'rewards': self.total_rewards[arm],
                'empirical_ctr': empirical_mean,
                'estimated_ctr': theoretical_mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
        return stats
```

## Production Architecture

### Real-Time Decision Service

We built a low-latency decision service that could select recommendation algorithms in real-time:

```python
class RecommendationBanditService:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('bandit-state')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # Initialize bandits for different user segments
        self.bandits = {
            'new_users': ThompsonSamplingMAB(['collab_filter', 'content_based', 'hybrid']),
            'returning_users': ThompsonSamplingMAB(['collab_filter', 'content_based', 'hybrid']),
            'premium_users': ThompsonSamplingMAB(['collab_filter', 'content_based', 'hybrid'])
        }
        
        self.load_bandit_state()
        
    def get_recommendation_algorithm(self, user_id: str, user_segment: str) -> str:
        """Select algorithm for user recommendation"""
        bandit = self.bandits.get(user_segment, self.bandits['returning_users'])
        selected_arm = bandit.select_arm()
        
        # Log selection for tracking
        self.log_arm_selection(user_id, user_segment, selected_arm)
        
        return selected_arm
        
    def record_outcome(self, user_id: str, user_segment: str, 
                      algorithm: str, clicked: bool, purchased: bool):
        """Update bandit with user interaction outcome"""
        # Define reward function (can be customized)
        if purchased:
            reward = 1.0  # Purchase is most valuable
        elif clicked:
            reward = 0.1  # Click has some value
        else:
            reward = 0.0  # No interaction
            
        bandit = self.bandits.get(user_segment, self.bandits['returning_users'])
        bandit.update(algorithm, reward)
        
        # Persist state
        self.save_bandit_state(user_segment)
        
        # Send metrics to CloudWatch
        self.send_metrics(user_segment, algorithm, reward)
        
    def load_bandit_state(self):
        """Load bandit state from DynamoDB"""
        try:
            for segment in self.bandits:
                response = self.table.get_item(Key={'segment': segment})
                if 'Item' in response:
                    state = response['Item']['state']
                    bandit = self.bandits[segment]
                    bandit.alpha = state['alpha']
                    bandit.beta = state['beta']
                    bandit.total_pulls = state['total_pulls']
                    bandit.total_rewards = state['total_rewards']
        except Exception as e:
            print(f"Error loading bandit state: {e}")
            
    def save_bandit_state(self, segment: str):
        """Save bandit state to DynamoDB"""
        bandit = self.bandits[segment]
        state = {
            'alpha': bandit.alpha,
            'beta': bandit.beta,
            'total_pulls': bandit.total_pulls,
            'total_rewards': bandit.total_rewards
        }
        
        self.table.put_item(
            Item={
                'segment': segment,
                'state': state,
                'updated_at': int(time.time())
            }
        )
```

### Integration with Recommendation Engine

```python
class RecommendationEngine:
    def __init__(self):
        self.bandit_service = RecommendationBanditService()
        self.algorithms = {
            'collab_filter': CollaborativeFilteringModel(),
            'content_based': ContentBasedModel(),
            'hybrid': HybridModel()
        }
        
    def get_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        # Determine user segment
        user_segment = self.classify_user_segment(user_id)
        
        # Select algorithm using bandit
        algorithm = self.bandit_service.get_recommendation_algorithm(user_id, user_segment)
        
        # Generate recommendations
        model = self.algorithms[algorithm]
        recommendations = model.predict(user_id, num_recommendations)
        
        # Add tracking metadata
        for rec in recommendations:
            rec['algorithm_used'] = algorithm
            rec['user_segment'] = user_segment
            rec['timestamp'] = time.time()
            
        return recommendations
        
    def record_user_interaction(self, user_id: str, item_id: str, 
                               interaction_type: str, metadata: Dict):
        """Record user interaction for bandit learning"""
        algorithm = metadata.get('algorithm_used')
        user_segment = metadata.get('user_segment')
        
        if algorithm and user_segment:
            clicked = interaction_type in ['click', 'view']
            purchased = interaction_type == 'purchase'
            
            self.bandit_service.record_outcome(
                user_id, user_segment, algorithm, clicked, purchased
            )
```

## Advanced Features

### Context-Aware Bandits

We extended the basic MAB to include contextual information:

```python
class ContextualThompsonSampling:
    def __init__(self, arms: List[str], context_dim: int):
        self.arms = arms
        self.context_dim = context_dim
        
        # Linear Thompson Sampling parameters
        self.A = {arm: np.eye(context_dim) for arm in arms}  # Precision matrix
        self.b = {arm: np.zeros(context_dim) for arm in arms}  # Reward vector
        self.alpha = 1.0  # Noise parameter
        
    def select_arm(self, context: np.ndarray) -> str:
        """Select arm based on context"""
        samples = {}
        
        for arm in self.arms:
            # Posterior mean
            A_inv = np.linalg.inv(self.A[arm])
            theta_hat = A_inv.dot(self.b[arm])
            
            # Sample from posterior
            theta_sample = np.random.multivariate_normal(
                theta_hat, self.alpha * A_inv
            )
            
            # Expected reward for this context
            samples[arm] = context.dot(theta_sample)
            
        return max(samples.keys(), key=lambda k: samples[k])
        
    def update(self, arm: str, context: np.ndarray, reward: float):
        """Update arm parameters"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

### Multi-Objective Optimization

We optimized for multiple objectives simultaneously:

```python
class MultiObjectiveMAB:
    def __init__(self, arms: List[str], objectives: List[str], weights: Dict[str, float]):
        self.arms = arms
        self.objectives = objectives
        self.weights = weights
        
        # Separate bandit for each objective
        self.bandits = {
            obj: ThompsonSamplingMAB(arms) for obj in objectives
        }
        
    def select_arm(self) -> str:
        """Select arm optimizing weighted combination of objectives"""
        arm_scores = {arm: 0.0 for arm in self.arms}
        
        for objective in self.objectives:
            # Get samples from each objective's bandit
            bandit = self.bandits[objective]
            samples = {}
            
            for arm in self.arms:
                samples[arm] = np.random.beta(bandit.alpha[arm], bandit.beta[arm])
                
            # Weight by objective importance
            weight = self.weights.get(objective, 1.0)
            for arm in self.arms:
                arm_scores[arm] += weight * samples[arm]
                
        return max(arm_scores.keys(), key=lambda k: arm_scores[k])
        
    def update(self, arm: str, rewards: Dict[str, float]):
        """Update with multi-objective rewards"""
        for objective, reward in rewards.items():
            if objective in self.bandits:
                self.bandits[objective].update(arm, reward)
```


### Traffic Allocation Evolution

The bandit learned to allocate traffic dynamically, gradually shifting from equal exploration to favoring the best-performing algorithm while maintaining some exploration of alternatives.

### Segment-Specific Insights

Different user segments showed different optimal algorithms:

- **New users**: Content-based performed best (cold-start problem)
- **Returning users**: Hybrid approach dominated
- **Premium users**: Collaborative filtering with premium data worked well

## Monitoring and Observability

### Real-Time Dashboards

```python
class BanditMonitoringDashboard:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        
    def create_bandit_metrics_dashboard(self):
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["RecommendationBandit", "ArmPulls", "Algorithm", "collab_filter"],
                            [".", ".", ".", "content_based"],
                            [".", ".", ".", "hybrid"]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": "us-west-2",
                        "title": "Algorithm Selection Frequency"
                    }
                },
                {
                    "type": "metric", 
                    "properties": {
                        "metrics": [
                            ["RecommendationBandit", "ConversionRate", "Algorithm", "collab_filter"],
                            [".", ".", ".", "content_based"],
                            [".", ".", ".", "hybrid"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-west-2",
                        "title": "Conversion Rate by Algorithm"
                    }
                }
            ]
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName='RecommendationBanditMonitoring',
            DashboardBody=json.dumps(dashboard_body)
        )
```

### Automated Alerts

```yaml
# CloudWatch alarms for bandit performance
BanditPerformanceAlerts:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: BanditRegretTooHigh
    AlarmDescription: Bandit regret exceeding acceptable threshold
    MetricName: CumulativeRegret
    Namespace: RecommendationBandit
    Statistic: Average
    Period: 3600
    EvaluationPeriods: 2
    Threshold: 0.1
    ComparisonOperator: GreaterThanThreshold
    AlarmActions:
      - !Ref BanditAlertsSnsTopic
```

## Challenges and Solutions

### 1. Cold Start Problem
- **Challenge**: New algorithms have no performance data
- **Solution**: Optimistic initialization with higher prior confidence

### 2. Concept Drift
- **Challenge**: User preferences change over time
- **Solution**: Exponential forgetting and sliding window updates

### 3. Statistical Significance
- **Challenge**: Ensuring decisions are statistically sound
- **Solution**: Confidence intervals and regret bounds monitoring

## Advanced Techniques

### 1. Bandit with Early Stopping
```python
class EarlyStoppingMAB(ThompsonSamplingMAB):
    def __init__(self, *args, confidence_level=0.95, min_samples=100):
        super().__init__(*args)
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        
    def check_early_stopping(self) -> Tuple[bool, str]:
        """Check if we can stop and declare a winner"""
        if min(self.total_pulls.values()) < self.min_samples:
            return False, "Insufficient samples"
            
        # Calculate confidence intervals
        arm_stats = self.get_arm_statistics()
        best_arm = max(arm_stats.keys(), key=lambda k: arm_stats[k]['estimated_ctr'])
        
        # Check if best arm's CI is above all others
        for arm in self.arms:
            if arm != best_arm:
                if arm_stats[best_arm]['ci_lower'] <= arm_stats[arm]['ci_upper']:
                    return False, "Confidence intervals overlap"
                    
        return True, best_arm
```

### 2. Budget-Constrained Bandits
```python
class BudgetConstrainedMAB:
    def __init__(self, arms: List[str], total_budget: float, arm_costs: Dict[str, float]):
        self.arms = arms
        self.remaining_budget = total_budget
        self.arm_costs = arm_costs
        self.bandit = ThompsonSamplingMAB(arms)
        
    def select_arm(self) -> str:
        """Select arm considering budget constraints"""
        affordable_arms = [
            arm for arm in self.arms 
            if self.arm_costs[arm] <= self.remaining_budget
        ]
        
        if not affordable_arms:
            return None  # Out of budget
            
        # Select from affordable arms
        temp_bandit = ThompsonSamplingMAB(affordable_arms)
        temp_bandit.alpha = {arm: self.bandit.alpha[arm] for arm in affordable_arms}
        temp_bandit.beta = {arm: self.bandit.beta[arm] for arm in affordable_arms}
        
        return temp_bandit.select_arm()
        
    def update(self, arm: str, reward: float):
        """Update bandit and budget"""
        self.bandit.update(arm, reward)
        self.remaining_budget -= self.arm_costs[arm]
```

## Key Takeaways

1. **Adaptive beats static**: MAB significantly outperformed fixed A/B testing
2. **Context matters**: User segmentation improved performance substantially  
3. **Multi-objective is realistic**: Real systems optimize for multiple metrics
4. **Monitoring is critical**: Real-time visibility enables quick adjustments
5. **Business alignment**: Success metrics must align with business objectives

## Future Enhancements

We're exploring several advanced directions:

- **Deep contextual bandits** using neural networks
- **Federated bandits** across multiple product categories
- **Adversarial bandits** robust to manipulation
- **Hierarchical bandits** for recommendation strategy selection

Multi-armed bandits transformed our recommendation system from a static, slow-adapting system into a dynamic, learning platform that continuously optimizes for user satisfaction and business metrics. The key insight: in a world where user preferences and market conditions change rapidly, your algorithms should adapt just as quickly.

Thompson sampling gave us the mathematical foundation to balance exploration and exploitation optimally, while the production architecture ensured we could operate at scale with the reliability our business demanded.