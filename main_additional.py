from functions_additional import *

def main():
    """
    Main execution pipeline for bank-additional dataset analysis
    """
    print("🏦 BANK MARKETING ANALYSIS - BANK-ADDITIONAL DATASET")
    print("=" * 80)

    # 1. Load the data
    print("\n📁 STEP 1: LOADING DATA")
    print("-" * 50)
    df = load_bank_additional_data('bank-additional-full.csv')

    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return

    # 2. Explore the data
    print("\n🔍 STEP 2: DATA EXPLORATION")
    print("-" * 50)
    exploration_results = explore_bank_additional_data(df)

    # 3. Visualize the data
    print("\n📊 STEP 3: DATA VISUALIZATION")
    print("-" * 50)
    print("Generating comprehensive visualizations...")
    visualize_bank_additional_data(df)

    # 4. Preprocess data WITH duration (benchmark model)
    print("\n🔧 STEP 4A: PREPROCESSING WITH DURATION")
    print("-" * 50)
    X_train_with, X_val_with, X_test_with, y_train_with, y_val_with, y_test_with, features_with, preprocessing_info_with = preprocess_bank_additional(
        df, include_duration=True, random_state=42
    )

    # Display preprocessing summary
    get_preprocessing_summary_additional(preprocessing_info_with)

    # 5. Preprocess data WITHOUT duration (realistic model)
    print("\n🔧 STEP 4B: PREPROCESSING WITHOUT DURATION")
    print("-" * 50)
    X_train_no, X_val_no, X_test_no, y_train_no, y_val_no, y_test_no, features_no, preprocessing_info_no = preprocess_bank_additional(
        df, include_duration=False, random_state=42
    )

    # Display preprocessing summary
    get_preprocessing_summary_additional(preprocessing_info_no)

    print("\n🤖 TRAINING LOGISTIC REGRESSION (WITH duration)")
    model_with, val_acc_with, test_acc_with = train_evaluate_logistic_regression(
        X_train_with, y_train_with, X_val_with, y_val_with, X_test_with, y_test_with
    )

    # 6. Train and evaluate Logistic Regression WITHOUT duration
    print("\n🤖 TRAINING LOGISTIC REGRESSION (WITHOUT duration)")
    model_no, val_acc_no, test_acc_no = train_evaluate_logistic_regression(
        X_train_no, y_train_no, X_val_no, y_val_no, X_test_no, y_test_no
    )

    # 9. Train and evaluate Decision Tree WITH duration
    print("\n🤖 TRAINING DECISION TREE (WITH duration)")
    dt_model_with, dt_val_acc_with, dt_test_acc_with = train_evaluate_decision_tree(
        X_train_with, y_train_with, X_val_with, y_val_with, X_test_with, y_test_with
    )
    # 10. Train and evaluate Decision Tree WITHOUT duration
    print("\n🤖 TRAINING DECISION TREE (WITHOUT duration)")
    dt_model_no, dt_val_acc_no, dt_test_acc_no = train_evaluate_decision_tree(
        X_train_no, y_train_no, X_val_no, y_val_no, X_test_no, y_test_no
    )


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

    # 7. Key insights and comparison with bank-full
    print("\n💡 STEP 6: KEY INSIGHTS & ADVANTAGES")
    print("-" * 50)
    print(f"• Dataset has {df.shape[0]} samples and {df.shape[1] - 1} features")
    print(f"• Target class imbalance ratio: {exploration_results['class_imbalance_ratio']:.2f}")

    # Unknown values summary
    if exploration_results['unknown_values']:
        print(f"• 'Unknown' values found in {len(exploration_results['unknown_values'])} features:")
        for feature, info in exploration_results['unknown_values'].items():
            print(f"  - {feature}: {info['percentage']:.1f}%")
    else:
        print("• No 'unknown' values found")

    # Economic features insights
    economic_features = exploration_results['economic_features']
    print(f"• Economic indicators: {len(economic_features)} features")
    print(f"  Features: {economic_features}")

    # Correlation insights
    if 'high_corr_pairs' in exploration_results and exploration_results['high_corr_pairs']:
        print(f"• Highly correlated economic features detected:")
        for feat1, feat2, corr in exploration_results['high_corr_pairs']:
            print(f"  - {feat1} ↔ {feat2}: {corr:.3f}")

    # Feature comparison with bank-full
    print(f"\n🔄 COMPARISON WITH BANK-FULL:")
    print(f"• Additional features: 5 economic indicators + refined categorical features")
    print(f"• More granular education categories (8 vs 4)")
    print(f"• Day of week instead of day of month")
    print(f"• 'Unknown' values handled as valid categories")

    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if exploration_results['class_imbalance_ratio'] > 2:
        print("• Consider using SMOTE or class weights for imbalanced data")

    if 'high_corr_pairs' in exploration_results and exploration_results['high_corr_pairs']:
        print("• Monitor highly correlated economic features for multicollinearity")

    print("• Economic features likely to be highly predictive")
    print("• Larger feature space may require regularization")

    print("\n✅ DATA LOADING AND PREPROCESSING COMPLETED!")
    print("Ready for advanced model training phase...")

    # Store processed datasets for model training
    datasets = {
        'with_duration': {
            'X_train': X_train_with,
            'X_val': X_val_with,
            'X_test': X_test_with,
            'y_train': y_train_with,
            'y_val': y_val_with,
            'y_test': y_test_with,
            'features': features_with,
            'preprocessing_info': preprocessing_info_with
        },
        'without_duration': {
            'X_train': X_train_no,
            'X_val': X_val_no,
            'X_test': X_test_no,
            'y_train': y_train_no,
            'y_val': y_val_no,
            'y_test': y_test_no,
            'features': features_no,
            'preprocessing_info': preprocessing_info_no
        }
    }

    return datasets, exploration_results


if __name__ == "__main__":
    # Execute main pipeline
    processed_datasets, exploration_summary = main()

    # Display final summary
    print("\n" + "=" * 80)
    print("🎊 BANK-ADDITIONAL DATASET READY FOR MODEL TRAINING!")
    print("=" * 80)
    print(f"✅ Datasets prepared: 2 variants (with/without duration)")
    print(f"✅ Features engineered: Economic indicators + refined categoricals")
    print(f"✅ Data quality: 'Unknown' values handled appropriately")
    print(f"✅ Scaling applied: Separate scaling for client vs economic features")
    print("💾 Processed datasets available in 'processed_datasets' dictionary")