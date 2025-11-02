🧠 Data Science Project Portfolio
This portfolio showcases three end-to-end data science projects built from real-world banking analytics experience. Each project demonstrates how machine learning, SQL, and business logic can work together to solve challenges in customer segmentation, cross-selling, and behavioral prediction.

All code snippets have been sanitized, refactored, and rewritten to remove confidential data and preserve only the analytical and modeling logic.

📂 Projects
1. Detect Loan Intent from Mobile App Behavioral Sequences
Goal:
Leverage customer digital footprints to predict loan intent in the absence of rich transactional data.

Highlights:
Developed an LSTM-based sequential model to learn behavioral patterns from clickstream data.
Integrated page embeddings, dwell time, and action count with padded sessions for variable sequence lengths.
Utilized bidirectional LSTM, attention mechanism, and MLP fusion with static features for improved accuracy.
Implemented GPU-optimized training with autocast and GradScaler for efficient large-scale experimentation.

2. Cross-sell Investment Products to Mortgage Clients
Goal:
Enable post-mortgage cross-selling of investment products through predictive modeling and automation.

Highlights:
Designed within the Customer Value Optimization (CVO) framework across five stages: Identify, Engage, Convert, Maximize, and Track.
Built two XGBoost propensity models to identify affluent-potential clients during and after mortgage application.
Automated data pipelines and lead delivery to frontline CRM and mail systems.
Developed a Tableau dashboard to monitor conversions, asset growth, and consultant performance.

3. Multi-Model Cross-Selling Engine for Personal Loan
Goal:
Identify high-potential customers for personal loans through segmentation and predictive modeling.

Highlights:
Applied K-Means clustering to group existing customers into nine segments based on product holdings and transaction behaviors.
Built segment-specific XGBoost classifiers to predict loan propensity.
Employed PCA for dimensionality reduction and Optuna for hyperparameter optimization.
Integrated lead scoring outputs into telemarketing workflows and mobile app notifications to boost conversion rates.

🧩 Tech Stack
Languages: Python, SQL
Libraries & Tools: XGBoost, Optuna, SHAP, PyTorch, Pandas, NumPy, Scikit-learn
Visualization: Tableau, Matplotlib, Seaborn
Deployment & Workflow: Git, CRM integration, Automated notifications

🔒 Disclaimer
All datasets and code snippets in this repository have been sanitized and reconstructed for demonstration purposes.
They reflect only the analytical logic, not actual business data or confidential systems.