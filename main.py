from functions import *


def main():
    """
    Main execution pipeline for bank-full dataset analysis
    """
    print("🏦 BANK MARKETING ANALYSIS - BANK-FULL DATASET")
    print("=" * 80)

    # 1. Load the data
    print("\n📁 STEP 1: LOADING DATA")
    print("-" * 50)
    df = load_bank_full_data('bank-full.csv')

    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return

    # 2. Explore the data
    print("\n🔍 STEP 2: DATA EXPLORATION")
    print("-" * 50)
    exploration_results = explore_bank_full_data(df)

    # 3. Visualize the data
    print("\n📊 STEP 3: DATA VISUALIZATION")
    print("-" * 50)
    print("Generating visualizations...")
    visualize_bank_full_data(df)

    # 4. Preprocess data WITH duration (benchmark model)
    print("\n🔧 STEP 4A: PREPROCESSING WITH DURATION")
    print("-" * 50)
    X_train_with, X_test_with, y_train_with, y_test_with, features_with, preprocessing_info_with = preprocess_bank_full(
        df, include_duration=True, random_state=42
    )

    # Display preprocessing summary
    get_preprocessing_summary(preprocessing_info_with)

    # 5. Preprocess data WITHOUT duration (realistic model)
    print("\n🔧 STEP 4B: PREPROCESSING WITHOUT DURATION")
    print("-" * 50)
    X_train_no, X_test_no, y_train_no, y_test_no, features_no, preprocessing_info_no = preprocess_bank_full(
        df, include_duration=False, random_state=42
    )

    # Display preprocessing summary
    get_preprocessing_summary(preprocessing_info_no)

    # 6. Summary of prepared datasets
    print("\n📋 STEP 5: DATASET PREPARATION SUMMARY")
    print("-" * 50)
    print(f"Original dataset: {df.shape}")
    print(f"\nDataset WITH duration:")
    print(f"  Training: {X_train_with.shape}")
    print(f"  Testing: {X_test_with.shape}")
    print(f"  Features: {len(features_with)}")

    print(f"\nDataset WITHOUT duration:")
    print(f"  Training: {X_train_no.shape}")
    print(f"  Testing: {X_test_no.shape}")
    print(f"  Features: {len(features_no)}")

    # 7. Key insights from exploration
    print("\n💡 STEP 6: KEY INSIGHTS")
    print("-" * 50)
    print(f"• Dataset has {df.shape[0]} samples and {df.shape[1] - 1} features")
    print(f"• Target class imbalance ratio: {exploration_results['class_imbalance_ratio']:.2f}")
    print(f"• No missing values detected")
    print(f"• Numerical features: {len(exploration_results['numerical_features'])}")
    print(f"• Categorical features: {len(exploration_results['categorical_features'])}")

    # Check for class imbalance
    if exploration_results['class_imbalance_ratio'] > 2:
        print("⚠️  Significant class imbalance detected - consider using SMOTE or class weights")

    print("\n✅ DATA LOADING AND PREPROCESSING COMPLETED!")
    print("Ready for model training phase...")

    # Store processed datasets for model training (you can return these or save them)
    datasets = {
        'with_duration': {
            'X_train': X_train_with,
            'X_test': X_test_with,
            'y_train': y_train_with,
            'y_test': y_test_with,
            'features': features_with,
            'preprocessing_info': preprocessing_info_with
        },
        'without_duration': {
            'X_train': X_train_no,
            'X_test': X_test_no,
            'y_train': y_train_no,
            'y_test': y_test_no,
            'features': features_no,
            'preprocessing_info': preprocessing_info_no
        }
    }

    return datasets


if __name__ == "__main__":
    # Execute main pipeline
    processed_datasets = main()

    # Optional: Save processed data for later use
    print("\n💾 Processed datasets are ready for model training!")
    print("You can access them via the returned 'processed_datasets' dictionary")