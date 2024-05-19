# Ecommerce-Product-Categorization

# Project Overview
In the rapidly evolving world of eCommerce, accurate product categorization is crucial for ensuring seamless customer experiences, reducing search friction, and increasing product discoverability. This project aims to develop a multi-class text classifier that categorizes products with maximum accuracy based on their descriptions and other features.

# Dataset
The dataset consists of detailed product information, including product descriptions, names, prices, brands, and categories. The training set contains 14,999 entries, and the test set contains 2,534 entries. Each entry includes features such as:

product_name
description
brand
retail_price
discounted_price
product_category_tree (target variable)
Data Preprocessing
To ensure the integrity of our dataset, we handled missing values by dropping entries without product_category_tree and description data. This step was crucial for maintaining the quality and relevance of the data used for model training.

# Feature Engineering
Text Vectorization: I have converted product descriptions into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency), limiting the vocabulary to the top 5000 features. This transformation captures the importance of words in each description relative to the entire dataset.

Label Encoding: I have transformed the target labels, product_category_tree, from categorical text into numerical format using LabelEncoder, facilitating compatibility with machine learning algorithms.

# Model Development
Handling Class Imbalance: I have used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples for underrepresented categories, resulting in a balanced dataset.

# Model Training
I have trained a Logistic Regression model with a maximum of 1000 iterations and balanced class weights to address any residual class imbalance. This approach ensures the model is well-trained on a diverse and representative sample, improving its ability to accurately categorize products.

# Model Evaluation
Performance Metrics: The model achieved a training accuracy of 0.99, indicating high accuracy in learning from the training data.

Classification Report: Detailed precision, recall, and F1-score metrics for each category were calculated, providing insights into the model's performance across different classes.

Confusion Matrix: A confusion matrix heatmap was used to visualize the model's classification performance, highlighting areas of strength and where misclassifications occurred.

# Results and Findings
The model demonstrated high accuracy, effectively distinguishing between product categories.
SMOTE was instrumental in balancing the training data, which helped in improving the model's performance.
The confusion matrix and classification report provided valuable insights for further refinement.
# Future Work
Enhanced Feature Engineering: Explore advanced techniques such as word embeddings or topic modeling.
Alternative Algorithms: Experiment with other machine learning algorithms like Random Forests or neural networks.
Model Deployment: Deploy the model into a production environment to automate product categorization tasks.
Continuous Monitoring: Implement a feedback loop for continuous model improvement based on real-time data.
# Lessons Learned
Balancing model complexity with practical implementation is crucial for efficiency.
Integrating diverse features requires careful preprocessing.
Continuous evaluation and experimentation are key to refining model performance.
# Conclusion
This project showcases the development of a robust framework for product categorization in eCommerce, leveraging text classification techniques to enhance product discoverability and improve customer experience. The insights and methodologies developed here lay the foundation for future enhancements and practical applications in the eCommerce industry.

