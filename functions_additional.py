import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


def load_bank_additional_data(filepath='bank-additional-full.csv'):
    try:
        # Bank datasets typically use semicolon separator
        df = pd.read_csv(filepath, sep=';')
        print(f"✅ Bank-Additional data loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def explore_bank_additional_data(df):
    """
    Comprehensive data exploration for bank-additional dataset

    Parameters:
    df (DataFrame): The bank-additional dataset

    Returns:
    dict: Summary statistics and insights
    """
    print("=" * 60)
    print("📊 BANK-ADDITIONAL DATASET EXPLORATION")
    print("=" * 60)

    exploration_results = {}

    # 1. Basic Dataset Info
    print("\n1️⃣ BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")  # Excluding target
    print(f"Number of samples: {df.shape[0]}")

    # 2. Data types
    print("\n2️⃣ DATA TYPES")
    print("-" * 40)
    print(df.dtypes)

    # 3. Missing values and 'unknown' analysis
    print("\n3️⃣ MISSING VALUES & 'UNKNOWN' ANALYSIS")
    print("-" * 40)
    missing_values = df.isnull().sum()
    print("True missing values (NaN) per column:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No NaN values found")

    # Detailed 'unknown' values analysis for bank-additional
    print("\n'Unknown' values analysis:")
    unknown_summary = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unknown_count = (df[col] == 'unknown').sum()
            if unknown_count > 0:
                unknown_pct = unknown_count / len(df) * 100
                print(f"{col}: {unknown_count} ({unknown_pct:.2f}%)")
                unknown_summary[col] = {'count': unknown_count, 'percentage': unknown_pct}

    exploration_results['unknown_values'] = unknown_summary

    # 4. Target variable analysis
    print("\n4️⃣ TARGET VARIABLE ANALYSIS")
    print("-" * 40)
    target_dist = df['y'].value_counts()
    target_pct = df['y'].value_counts(normalize=True) * 100

    print("Target distribution:")
    for val, count in target_dist.items():
        pct = target_pct[val]
        print(f"  {val}: {count} ({pct:.2f}%)")

    exploration_results['target_distribution'] = target_dist
    exploration_results['class_imbalance_ratio'] = target_dist.max() / target_dist.min()

    # 5. Numerical features analysis (including economic indicators)
    print("\n5️⃣ NUMERICAL FEATURES ANALYSIS")
    print("-" * 40)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Separate economic indicators
    economic_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    client_features = [col for col in numerical_cols if col not in economic_features]

    print(f"Client numerical features ({len(client_features)}): {client_features}")
    print(f"Economic indicators ({len(economic_features)}): {economic_features}")

    print("\nClient features statistics:")
    if client_features:
        print(df[client_features].describe())

    print("\nEconomic indicators statistics:")
    if economic_features:
        print(df[economic_features].describe())

    exploration_results['numerical_features'] = numerical_cols
    exploration_results['economic_features'] = economic_features
    exploration_results['client_features'] = client_features

    # 6. Categorical features analysis
    print("\n6️⃣ CATEGORICAL FEATURES ANALYSIS")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')  # Remove target from features

    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    for col in categorical_cols:
        unique_vals = df[col].nunique()
        print(f"\n{col}: {unique_vals} unique values")
        print(f"  Values: {df[col].unique()}")
        value_counts = df[col].value_counts()
        print(f"  Distribution:\n{value_counts}")

        # Check for class imbalance in categorical features
        if unique_vals > 1:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 10:
                print(f"  ⚠️  High imbalance ratio: {imbalance_ratio:.2f}")

    exploration_results['categorical_features'] = categorical_cols

    # 7. Economic indicators correlation
    print("\n7️⃣ ECONOMIC INDICATORS CORRELATION")
    print("-" * 40)
    if economic_features:
        econ_corr = df[economic_features].corr()
        print("Economic indicators correlation matrix:")
        print(econ_corr.round(3))

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(econ_corr.columns)):
            for j in range(i + 1, len(econ_corr.columns)):
                corr_val = abs(econ_corr.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((econ_corr.columns[i], econ_corr.columns[j], corr_val))

        if high_corr_pairs:
            print("\n⚠️  Highly correlated economic features (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} - {feat2}: {corr:.3f}")

        exploration_results['economic_correlation'] = econ_corr
        exploration_results['high_corr_pairs'] = high_corr_pairs

    return exploration_results


def visualize_bank_additional_data(df):
    """
    Create comprehensive visualizations for bank-additional dataset

    Parameters:
    df (pd.DataFrame): The bank-additional dataset
    """
    plt.style.use('default')

    # Create multiple figure sets for better organization

    # Figure 1: Basic distributions
    plt.figure(figsize=(20, 15))

    # 1. Target distribution
    plt.subplot(3, 4, 1)
    target_counts = df['y'].value_counts()
    colors = ['lightcoral', 'lightblue']
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    plt.title('Target Distribution (y)', fontsize=12, fontweight='bold')

    # 2. Age distribution
    plt.subplot(3, 4, 2)
    plt.hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Age Distribution', fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    # 3. Campaign distribution
    plt.subplot(3, 4, 3)
    plt.hist(df['campaign'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Campaign Contacts Distribution', fontweight='bold')
    plt.xlabel('Number of Contacts')
    plt.ylabel('Frequency')

    # 4. Duration distribution (log scale due to wide range)
    plt.subplot(3, 4, 4)
    duration_filtered = df[df['duration'] > 0]['duration']  # Remove zeros for log scale
    plt.hist(duration_filtered, bins=50, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Call Duration Distribution', fontweight='bold')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.yscale('log')

    # 5. Job distribution
    plt.subplot(3, 4, 5)
    job_counts = df['job'].value_counts()
    plt.barh(range(len(job_counts)), job_counts.values, color='lightsteelblue')
    plt.yticks(range(len(job_counts)), job_counts.index, fontsize=9)
    plt.title('Job Distribution', fontweight='bold')
    plt.xlabel('Count')

    # 6. Education distribution
    plt.subplot(3, 4, 6)
    education_counts = df['education'].value_counts()
    plt.barh(range(len(education_counts)), education_counts.values, color='lightcyan')
    plt.yticks(range(len(education_counts)), education_counts.index, fontsize=9)
    plt.title('Education Distribution', fontweight='bold')
    plt.xlabel('Count')

    # 7. Marital status
    plt.subplot(3, 4, 7)
    marital_counts = df['marital'].value_counts()
    plt.bar(marital_counts.index, marital_counts.values, color='plum', alpha=0.7)
    plt.title('Marital Status Distribution', fontweight='bold')
    plt.xlabel('Marital Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # 8. Contact method
    plt.subplot(3, 4, 8)
    contact_counts = df['contact'].value_counts()
    plt.bar(contact_counts.index, contact_counts.values, color='wheat', alpha=0.7)
    plt.title('Contact Method Distribution', fontweight='bold')
    plt.xlabel('Contact Type')
    plt.ylabel('Count')

    # 9. Day of week
    plt.subplot(3, 4, 9)
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    plt.bar(day_counts.index, day_counts.values, color='lightpink', alpha=0.7)
    plt.title('Contact Day of Week', fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # 10. Previous campaign outcome
    plt.subplot(3, 4, 10)
    poutcome_counts = df['poutcome'].value_counts()
    plt.bar(poutcome_counts.index, poutcome_counts.values, color='lightseagreen', alpha=0.7)
    plt.title('Previous Campaign Outcome', fontweight='bold')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # 11. Education vs Target
    plt.subplot(3, 4, 11)
    education_target = pd.crosstab(df['education'], df['y'], normalize='index') * 100
    education_target.plot(kind='bar', stacked=True, ax=plt.gca(), color=['lightcoral', 'lightblue'])
    plt.title('Education vs Target (%)', fontweight='bold')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.legend(title='Subscribed', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)

    # 12. Job vs Target (top 8 jobs)
    plt.subplot(3, 4, 12)
    top_jobs = df['job'].value_counts().head(8).index
    job_target = pd.crosstab(df[df['job'].isin(top_jobs)]['job'], df[df['job'].isin(top_jobs)]['y'],
                             normalize='index') * 100
    job_target.plot(kind='bar', stacked=True, ax=plt.gca(), color=['lightcoral', 'lightblue'])
    plt.title('Top 8 Jobs vs Target (%)', fontweight='bold')
    plt.xlabel('Job Type')
    plt.ylabel('Percentage')
    plt.legend(title='Subscribed', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Figure 2: Economic indicators
    economic_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    plt.figure(figsize=(20, 12))

    for i, feature in enumerate(economic_features, 1):
        # Distribution plots
        plt.subplot(2, 5, i)
        plt.hist(df[feature], bins=30, alpha=0.7, color='gold', edgecolor='black')
        plt.title(f'{feature} Distribution', fontweight='bold')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        # Box plots by target
        plt.subplot(2, 5, i + 5)
        df.boxplot(column=feature, by='y', ax=plt.gca())
        plt.title(f'{feature} by Target', fontweight='bold')
        plt.suptitle('')  # Remove automatic title

    plt.tight_layout()
    plt.show()

    # Figure 3: Correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # All numerical features correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_all = df[numerical_cols].corr()

    sns.heatmap(corr_all, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax1, fmt='.2f')
    ax1.set_title('All Numerical Features Correlation', fontsize=14, fontweight='bold')

    # Economic indicators correlation (zoomed in)
    if len(economic_features) > 1:
        corr_econ = df[economic_features].corr()
        sns.heatmap(corr_econ, annot=True, cmap='RdYlBu', center=0,
                    square=True, linewidths=0.5, ax=ax2, fmt='.3f')
        ax2.set_title('Economic Indicators Correlation', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def preprocess_bank_additional(df, include_duration=True, test_size=0.2, random_state=42):
    """
    Comprehensive preprocessing pipeline for bank-additional dataset

    Parameters:
    df (pd.DataFrame): Raw bank-additional dataset
    include_duration (bool): Whether to include duration feature
    test_size (float): Proportion of data for testing
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names, preprocessing_info)
    """
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING BANK-ADDITIONAL DATASET")
    print("=" * 60)

    # Make a copy to avoid modifying original data
    df_processed = df.copy()

    # 1. Handle duration feature
    if not include_duration and 'duration' in df_processed.columns:
        print("⚠️  Removing 'duration' feature (data leakage prevention)")
        df_processed = df_processed.drop('duration', axis=1)
    elif include_duration:
        print("⚠️  Keeping 'duration' feature (benchmark purposes)")

    # 2. Separate features and target
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 3. Encode target variable (yes=1, no=0)
    print("\n1️⃣ ENCODING TARGET VARIABLE")
    print("-" * 40)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
    print(f"Target encoding: {target_mapping}")

    # 4. Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Separate economic indicators for special handling
    economic_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    economic_features = [f for f in economic_features if f in numerical_features]
    client_numerical = [f for f in numerical_features if f not in economic_features]

    print(f"\n2️⃣ FEATURE IDENTIFICATION")
    print("-" * 40)
    print(f"Client numerical features ({len(client_numerical)}): {client_numerical}")
    print(f"Economic indicator features ({len(economic_features)}): {economic_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # 5. Handle categorical features (including 'unknown' values)
    print(f"\n3️⃣ HANDLING CATEGORICAL FEATURES")
    print("-" * 40)

    X_categorical = X[categorical_features]
    X_numerical = X[numerical_features]

    # Display 'unknown' handling strategy
    unknown_counts = {}
    for feature in categorical_features:
        unknown_count = (X_categorical[feature] == 'unknown').sum()
        if unknown_count > 0:
            unknown_pct = unknown_count / len(X_categorical) * 100
            unknown_counts[feature] = {'count': unknown_count, 'percentage': unknown_pct}
            print(f"\n{feature}: {unknown_count} 'unknown' values ({unknown_pct:.2f}%)")
            print(f"  Strategy: Treating 'unknown' as a valid category")

    # One-hot encode categorical features (this naturally handles 'unknown' as a category)
    X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True, prefix_sep='_')

    print(f"\nOriginal categorical features: {len(categorical_features)}")
    print(f"After one-hot encoding: {X_categorical_encoded.shape[1]} features")

    # Display encoding info for key features
    key_features = ['job', 'education', 'marital', 'default', 'housing', 'loan']
    for feature in key_features:
        if feature in categorical_features:
            original_categories = sorted(X[feature].unique())
            encoded_columns = [col for col in X_categorical_encoded.columns if col.startswith(f"{feature}_")]
            print(f"\n{feature}:")
            print(f"  Original categories ({len(original_categories)}): {original_categories}")
            print(f"  Encoded columns ({len(encoded_columns)}): {encoded_columns}")

    # 6. Handle numerical features with separate scaling for different types
    print(f"\n4️⃣ SCALING NUMERICAL FEATURES")
    print("-" * 40)

    # Scale client features and economic features separately (they have very different scales)
    client_scaler = StandardScaler()
    economic_scaler = StandardScaler()

    X_client_scaled = pd.DataFrame(
        client_scaler.fit_transform(X[client_numerical]),
        columns=client_numerical,
        index=X.index
    ) if client_numerical else pd.DataFrame(index=X.index)

    X_economic_scaled = pd.DataFrame(
        economic_scaler.fit_transform(X[economic_features]),
        columns=economic_features,
        index=X.index
    ) if economic_features else pd.DataFrame(index=X.index)

    print(f"Client features scaling: StandardScaler applied to {len(client_numerical)} features")
    print(f"Economic features scaling: StandardScaler applied to {len(economic_features)} features")

    if len(client_numerical) > 0:
        print(f"\nClient features statistics after scaling:")
        print(X_client_scaled.describe().round(3))

    if len(economic_features) > 0:
        print(f"\nEconomic features statistics after scaling:")
        print(X_economic_scaled.describe().round(3))

    # 7. Combine all processed features
    print(f"\n5️⃣ COMBINING FEATURES")
    print("-" * 40)

    # Combine scaled numerical and encoded categorical features
    feature_parts = []
    if not X_client_scaled.empty:
        feature_parts.append(X_client_scaled)
    if not X_economic_scaled.empty:
        feature_parts.append(X_economic_scaled)
    if not X_categorical_encoded.empty:
        feature_parts.append(X_categorical_encoded)

    X_processed = pd.concat(feature_parts, axis=1)
    feature_names = X_processed.columns.tolist()

    print(f"Final feature matrix shape: {X_processed.shape}")
    print(f"Total features: {len(feature_names)}")
    print(f"  - Client numerical: {len(client_numerical)}")
    print(f"  - Economic indicators: {len(economic_features)}")
    print(f"  - Categorical (encoded): {X_categorical_encoded.shape[1]}")

    # 8. Train-test split with stratification
    print(f"\n6️⃣ TRAIN-TEST SPLIT")
    print("-" * 40)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_processed, y_encoded,
        test_size=0.1,  # 10% for final testing
        random_state=random_state,
        stratify=y_encoded
    )

    # Second split: divide remaining 90% into train (80%) and validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.111,  # 10/90 = 0.111 (10% of total data)
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X_processed) * 100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / len(X_processed) * 100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X_processed) * 100:.1f}%)")
    print(f"Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Store comprehensive preprocessing information
    preprocessing_info = {
        'target_encoder': target_encoder,
        'client_scaler': client_scaler,
        'economic_scaler': economic_scaler,
        'client_numerical_features': client_numerical,
        'economic_features': economic_features,
        'categorical_features': categorical_features,
        'feature_names': feature_names,
        'include_duration': include_duration,
        'unknown_counts': unknown_counts,
        'encoding_info': {
            'categorical_encoded_features': X_categorical_encoded.columns.tolist(),
            'total_encoded_features': len(feature_names)
        }
    }

    print(f"\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessing_info


def get_preprocessing_summary_additional(preprocessing_info):
    """
    Display a comprehensive summary of preprocessing steps for bank-additional dataset

    Parameters:
    preprocessing_info (dict): Information from preprocessing pipeline
    """
    print("\n" + "=" * 60)
    print("📋 BANK-ADDITIONAL PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Duration feature included: {preprocessing_info['include_duration']}")
    print(f"Total features after preprocessing: {len(preprocessing_info['feature_names'])}")

    print(f"\n🔢 NUMERICAL FEATURES:")
    print(f"  Client features: {len(preprocessing_info['client_numerical_features'])}")
    print(f"    Features: {preprocessing_info['client_numerical_features']}")
    print(f"  Economic indicators: {len(preprocessing_info['economic_features'])}")
    print(f"    Features: {preprocessing_info['economic_features']}")

    print(f"\n📝 CATEGORICAL FEATURES:")
    print(f"  Original categorical: {len(preprocessing_info['categorical_features'])}")
    print(f"  After encoding: {len(preprocessing_info['encoding_info']['categorical_encoded_features'])}")
    print(f"  Features: {preprocessing_info['categorical_features']}")

    print(f"\n❓ 'UNKNOWN' VALUES HANDLING:")
    if preprocessing_info['unknown_counts']:
        for feature, info in preprocessing_info['unknown_counts'].items():
            print(f"  {feature}: {info['count']} unknown ({info['percentage']:.2f}%) - treated as valid category")
    else:
        print("  No 'unknown' values found")

    print(f"\n⚙️ PREPROCESSING METHODS:")
    print(f"  Target encoding: {dict(zip(preprocessing_info['target_encoder'].classes_, [0, 1]))}")
    print(f"  Client numerical scaling: StandardScaler")
    print(f"  Economic indicators scaling: StandardScaler (separate)")
    print(f"  Categorical encoding: One-hot encoding with drop_first=True")
    print(f"  Train-test split: Stratified to maintain class balance")


def train_evaluate_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains a Logistic Regression classifier and evaluates it on validation and test sets.
    Returns:
        model: Trained Logistic Regression model
        val_accuracy: Accuracy on the validation set
        test_accuracy: Accuracy on the test set
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return model, val_accuracy, test_accuracy


def train_evaluate_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains a simple Decision Tree classifier and evaluates it on validation and test sets.
    Returns:
        model: Trained Decision Tree model
        val_accuracy: Accuracy on the validation set
        test_accuracy: Accuracy on the test set
    """
    # Keep the tree small/simple for speed and baseline
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"Validation Accuracy (Decision Tree): {val_accuracy:.4f}")
    print(f"Test Accuracy (Decision Tree): {test_accuracy:.4f}")

    return model, val_accuracy, test_accuracy