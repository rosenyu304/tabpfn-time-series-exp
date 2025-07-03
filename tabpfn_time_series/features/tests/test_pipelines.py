import pandas as pd


def test_train_test_split_time_series(train_df, train_tsdf):
    """
    Asserts that the key columns in train_df and the DataFrame version
    of train_tsdf are identical.
    """
    # Convert tsdf to a DataFrame once to avoid repeated computation
    train_tsdf_df = train_tsdf.to_data_frame().reset_index()

    # Use assert directly for testing. The message will only be shown on failure.
    pd.testing.assert_series_equal(
        train_df["target"],
        train_tsdf_df["target"],
        check_names=False,
        obj="Target Series",
    )
    pd.testing.assert_series_equal(
        train_df["timestamp"],
        train_tsdf_df["timestamp"],
        check_names=False,
        obj="Timestamp Series",
    )
    pd.testing.assert_series_equal(
        train_df["item_id"],
        train_tsdf_df["item_id"],
        check_names=False,
        obj="Item ID Series",
    )


def test_running_index_feature(train_df, train_tsdf):
    """
    Asserts that the 'running_index' column is the same in both data structures.
    """
    train_tsdf_running_index = train_tsdf.to_data_frame().reset_index()["running_index"]

    # Replace the if/else block with a single, clear assertion
    pd.testing.assert_series_equal(
        train_df["running_index"],
        train_tsdf_running_index,
        check_names=False,
        obj="Running Index Series",
    )


def test_feature_transformer(train_df, train_tsdf_original):
    """
    Asserts that all common columns between train_df and train_tsdf_original
    match for all item_ids.
    """
    # Step 1: Convert and set index for comparison
    df_original = train_tsdf_original.to_data_frame().reset_index()
    df1 = train_df.set_index(["item_id", "timestamp"]).sort_index()
    df2 = df_original.set_index(["item_id", "timestamp"]).sort_index()

    # Step 2: Find common columns to compare
    common_cols = df1.columns.intersection(df2.columns)

    # Step 3: Iterate and assert equality for each column and item
    # The original logic with a loop is good for pinpointing errors.
    # The assert statement correctly stops the test on the first failure.
    for col in common_cols:
        for item_id in df1.index.get_level_values("item_id").unique():
            s1 = df1.loc[item_id, col]
            s2 = df2.loc[item_id, col]

            # Use pandas' testing utility for a more informative comparison
            pd.testing.assert_series_equal(
                s1, s2, check_names=False, obj=f"Column '{col}' for item_id '{item_id}'"
            )
