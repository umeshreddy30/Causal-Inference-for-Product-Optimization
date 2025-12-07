import numpy as np
import pandas as pd
import logging

# Configure logging to look professional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Simulates a production-level dataset for a SaaS product experiment.
    Generates data with confounding variables to test causal inference capabilities.
    """
    def __init__(self, n_samples=10000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.df = None

    def generate_data(self):
        """
        Generates synthetic data with the following logic:
        - Treatment: 'used_new_feature' (Binary 0/1)
        - Outcome: 'total_spend' (Continuous $)
        - Confounder 1: 'account_age_months' (Older accounts spend more AND use features more)
        - Confounder 2: 'is_power_user' (Power users are biased towards treatment)
        """
        np.random.seed(self.seed)
        logging.info(f"Generating synthetic data for {self.n_samples} users...")

        # 1. Generate Confounders (The variables that bias the result)
        # Random account age between 1 and 60 months
        account_age = np.random.randint(1, 60, self.n_samples)
        
        # Power user status (30% of users)
        is_power_user = np.random.binomial(1, 0.3, self.n_samples)

        # 2. Assign Treatment (biased by confounders)
        # Probability of using the new feature increases with account age and power user status
        prob_treatment = (account_age / 100) + (is_power_user * 0.4)
        prob_treatment = np.clip(prob_treatment, 0, 1) # Ensure probability is between 0 and 1
        
        treatment = np.random.binomial(1, prob_treatment)

        # 3. Generate Outcome (Total Spend)
        # Base spend + Effect of Age + Effect of Power User + TRUE CAUSAL EFFECT (fixed at $10) + Noise
        true_causal_effect = 10 
        noise = np.random.normal(0, 5, self.n_samples)
        
        spend = (account_age * 0.5) + (is_power_user * 20) + (treatment * true_causal_effect) + noise

        # Create DataFrame
        self.df = pd.DataFrame({
            'account_age': account_age,
            'is_power_user': is_power_user,
            'used_new_feature': treatment,  # This is our 'Treatment'
            'total_spend': spend            # This is our 'Outcome'
        })
        
        logging.info("Data generation complete.")
        return self.df

    def save_data(self, filepath):
        if self.df is not None:
            self.df.to_csv(filepath, index=False)
            logging.info(f"Data saved to {filepath}")
        else:
            logging.error("No data to save. Run generate_data() first.")

if __name__ == "__main__":
    # Test the loader independently
    loader = DataLoader()
    df = loader.generate_data()
    loader.save_data("data/raw/experiment_data.csv")
    print(df.head())