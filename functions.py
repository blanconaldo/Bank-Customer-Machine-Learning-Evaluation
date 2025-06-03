import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')


def load_bank_full_data(filepath='bank-full.csv'):
    """
    Load the bank-full.csv dataset

    Parameters:
    filepath: Path to the bank-full.csv file

    Returns:
    pd.DataFrame: Loaded dataset
    """
    try:
        # Bank datasets typically use semicolon separator
        df = pd.read_csv(filepath, sep=';')
        print(f"✅ Data loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def explore_bank_full_data(df):
    """
    Comprehensive data exploration for bank-full dataset

    Parameters:
    df (pd.DataFrame): The bank-full dataset

    Returns:
    dict: Summary statistics and insights
    """
    print("=" * 60)
    print("📊 BANK-FULL DATASET EXPLORATION")
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

    # 3. Missing values
    print("\n3️⃣ MISSING VALUES CHECK")
    print("-" * 40)
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)

    # Check for 'unknown' values (treated as missing in some contexts)
    print("\n'Unknown' values per column:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unknown_count = (df[col] == 'unknown').sum()
            if unknown_count > 0:
                print(f"{col}: {unknown_count} ({unknown_count / len(df) * 100:.2f}%)")

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

    # 5. Numerical features analysis
    print("\n5️⃣ NUMERICAL FEATURES ANALYSIS")
    print("-" * 40)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical features: {numerical_cols}")
    print("\nNumerical features statistics:")
    print(df[numerical_cols].describe())

    # 6. Categorical features analysis
    print("\n6️⃣ CATEGORICAL FEATURES ANALYSIS")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')  # Remove target from features

    print(f"Categorical features: {categorical_cols}")
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        print(f"\n{col}: {unique_vals} unique values")
        print(f"  Values: {df[col].unique()[:10]}")  # Show first 10 values
        if unique_vals <= 10:  # Show distribution for features with few categories
            print(f"  Distribution:\n{df[col].value_counts()}")

    exploration_results['numerical_features'] = numerical_cols
    exploration_results['categorical_features'] = categorical_cols

    return exploration_results


def visualize_bank_full_data(df):
    """
    Create visualizations for bank-full dataset

    Parameters:
    df (pd.DataFrame): The bank-full dataset
    """
    plt.style.use('default')

    # 1. Target distribution
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    target_counts = df['y'].value_counts()
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Target Distribution (y)')

    # 2. Age distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    # 3. Balance distribution (handling outliers)
    plt.subplot(2, 3, 3)
    plt.hist(df['balance'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Balance Distribution')
    plt.xlabel('Balance (EUR)')
    plt.ylabel('Frequency')

    # 4. Job distribution
    plt.subplot(2, 3, 4)
    job_counts = df['job'].value_counts()
    plt.barh(range(len(job_counts)), job_counts.values)
    plt.yticks(range(len(job_counts)), job_counts.index)
    plt.title('Job Distribution')
    plt.xlabel('Count')

    # 5. Education vs Target
    plt.subplot(2, 3, 5)
    education_target = pd.crosstab(df['education'], df['y'], normalize='index') * 100
    education_target.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Education vs Target (Percentage)')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.legend(title='Subscribed')
    plt.xticks(rotation=45)

    # 6. Duration distribution
    plt.subplot(2, 3, 6)
    plt.hist(df['duration'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Call Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Correlation heatmap for numerical features
    plt.figure(figsize=(10, 8))
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = df[numerical_cols].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix - Numerical Features')
    plt.tight_layout()
    plt.show()


def preprocess_bank_full(df, include_duration=True, test_size=0.2, random_state=42):
    """
    Comprehensive preprocessing pipeline for bank-full dataset

    Parameters:
    df (pd.DataFrame): Raw bank-full dataset
    include_duration (bool): Whether to include duration feature
    test_size (float): Proportion of data for testing
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names, scalers_info)
    """
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING BANK-FULL DATASET")
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

    print(f"\n2️⃣ FEATURE IDENTIFICATION")
    print("-" * 40)
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # 5. Handle categorical features
    print(f"\n3️⃣ HANDLING CATEGORICAL FEATURES")
    print("-" * 40)

    # For bank-full, we'll use one-hot encoding for all categorical features
    X_categorical = X[categorical_features]
    X_numerical = X[numerical_features]

    # One-hot encode categorical features
    X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True, prefix_sep='_')

    print(f"Original categorical features: {len(categorical_features)}")
    print(f"After one-hot encoding: {X_categorical_encoded.shape[1]} features")

    # Display encoding info
    for feature in categorical_features:
        original_categories = X[feature].unique()
        encoded_columns = [col for col in X_categorical_encoded.columns if col.startswith(f"{feature}_")]
        print(f"\n{feature}:")
        print(f"  Original categories: {original_categories}")
        print(f"  Encoded columns: {encoded_columns}")

    # 6. Handle numerical features (scaling)
    print(f"\n4️⃣ SCALING NUMERICAL FEATURES")
    print("-" * 40)

    scaler = StandardScaler()
    X_numerical_scaled = pd.DataFrame(
        scaler.fit_transform(X_numerical),
        columns=numerical_features,
        index=X_numerical.index
    )

    print("Scaling applied: StandardScaler (mean=0, std=1)")
    print(f"Numerical features statistics after scaling:")
    print(X_numerical_scaled.describe().round(3))

    # 7. Combine processed features
    print(f"\n5️⃣ COMBINING FEATURES")
    print("-" * 40)

    X_processed = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
    feature_names = X_processed.columns.tolist()

    print(f"Final feature matrix shape: {X_processed.shape}")
    print(f"Total features: {len(feature_names)}")

    # 8. Train-test split
    print(f"\n6️⃣ TRAIN-TEST SPLIT")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded  # Maintain class distribution
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Store preprocessing information
    preprocessing_info = {
        'target_encoder': target_encoder,
        'numerical_scaler': scaler,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'feature_names': feature_names,
        'include_duration': include_duration
    }

    print(f"\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")

    return X_train, X_test, y_train, y_test, feature_names, preprocessing_info


def get_preprocessing_summary(preprocessing_info):
    """
    Display a summary of preprocessing steps applied

    Parameters:
    preprocessing_info (dict): Information from preprocessing pipeline
    """
    print("\n" + "=" * 60)
    print("📋 PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Duration feature included: {preprocessing_info['include_duration']}")
    print(f"Total features after preprocessing: {len(preprocessing_info['feature_names'])}")
    print(f"Numerical features: {len(preprocessing_info['numerical_features'])}")
    print(f"Categorical features: {len(preprocessing_info['categorical_features'])}")
    print(f"Target encoding: {dict(zip(preprocessing_info['target_encoder'].classes_, [0, 1]))}")
    print(f"Numerical scaling: StandardScaler applied")
    print(f"Categorical encoding: One-hot encoding with drop_first=True")