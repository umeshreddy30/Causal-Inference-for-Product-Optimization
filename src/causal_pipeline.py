import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CausalEngine")

class CausalIntelligenceEngine:
    """
    A production-grade pipeline for Causal Inference using DoWhy.
    Automates the end-to-end flow: Graph creation, Identification, Estimation, and Refutation.
    """
    
    def __init__(self, data, treatment_col, outcome_col, confounders):
        """
        Initialize the engine with dataset and causal definitions.
        
        :param data: Pandas DataFrame containing the experiment data.
        :param treatment_col: Name of the column representing the intervention (e.g., 'used_new_feature').
        :param outcome_col: Name of the column representing the result (e.g., 'total_spend').
        :param confounders: List of columns that influence both treatment and outcome (e.g., 'age').
        """
        self.df = data
        self.treatment = treatment_col
        self.outcome = outcome_col
        self.confounders = confounders
        self.model = None
        self.identified_estimand = None
        self.estimate = None

        # Ensure output directory exists for plots
        os.makedirs("notebooks/plots", exist_ok=True)

    def create_causal_graph(self):
        """
        Step 1: Model the problem.
        Defines the assumptions (the Causal Graph) linking Treatment, Outcome, and Confounders.
        """
        logger.info("Step 1: Building Causal Graph...")
        
        self.model = CausalModel(
            data=self.df,
            treatment=self.treatment,
            outcome=self.outcome,
            common_causes=self.confounders
        )
        
        logger.info("Causal Model initialized successfully.")
        
        # Try to save the graph visualization (Handles missing Graphviz binary gracefully)
        try:
            self.model.view_model(file_name="notebooks/plots/causal_graph", file_format="png")
            logger.info("Causal graph saved to notebooks/plots/causal_graph.png")
        except Exception as e:
            logger.warning(f"Could not save graph image (likely missing Graphviz binary): {e}")

    def identify_effect(self):
        """
        Step 2: Identify the Causal Effect.
        Uses the graph to find a statistical path to estimate the effect (e.g., Backdoor Criterion).
        """
        if self.model is None:
            raise ValueError("Model not built. Run create_causal_graph() first.")
            
        logger.info("Step 2: Identifying Causal Effect...")
        self.identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        
        # Log the identification strategy (Crucial for debugging)
        logger.info(f"Identified Estimand: {self.identified_estimand}")

    def estimate_effect(self, method="backdoor.propensity_score_matching"):
        """
        Step 3: Estimate the Effect.
        Calculates the actual numeric value of the impact.
        
        :param method: The statistical method to use. Default is Propensity Score Matching (PSM).
        """
        logger.info(f"Step 3: Estimating Effect using {method}...")
        
        self.estimate = self.model.estimate_effect(
            self.identified_estimand,
            method_name=method,
            target_units="ate" # Average Treatment Effect
        )
        
        logger.info(f"\n*** CAUSAL ESTIMATE ***\nMean Estimate: {self.estimate.value}")
        return self.estimate.value

    def validate_robustness(self):
        """
        Step 4: Refutation (The 'FAANG' Standard).
        Validates the result by challenging the assumptions.
        """
        logger.info("Step 4: Validating Robustness (Refutation Tests)...")
        
        # Test 1: Random Common Cause (Adds a random confounder; estimate should not change)
        refute_rcc = self.model.refute_estimate(
            self.identified_estimand,
            self.estimate,
            method_name="random_common_cause"
        )
        logger.info(f"Refutation (Random Common Cause): {refute_rcc}")

        # Test 2: Placebo Treatment (Replaces treatment with random noise; effect should go to 0)
        refute_placebo = self.model.refute_estimate(
            self.identified_estimand,
            self.estimate,
            method_name="placebo_treatment_refuter"
        )
        logger.info(f"Refutation (Placebo Treatment): {refute_placebo}")
        
        return refute_rcc, refute_placebo

# --- Execution Block for Testing ---
if __name__ == "__main__":
    # Load the data we generated in Hour 1
    try:
        df = pd.read_csv("data/raw/experiment_data.csv")
        
        # Initialize the engine
        engine = CausalIntelligenceEngine(
            data=df,
            treatment_col="used_new_feature",
            outcome_col="total_spend",
            confounders=["account_age", "is_power_user"]
        )
        
        # Run the Pipeline
        engine.create_causal_graph()
        engine.identify_effect()
        estimate = engine.estimate_effect()
        engine.validate_robustness()
        
        print(f"\nFinal Result: The new feature causes an increase of ${estimate:.2f} in spending.")
        
    except FileNotFoundError:
        logger.error("Data file not found. Please run src/data_loader.py first.")