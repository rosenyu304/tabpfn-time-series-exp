def test_train_test_split_time_series(train_df, train_tsdf):
    # check if the train_df and train_tsdf are the same
    if train_df["target"].equals(train_tsdf.to_data_frame().reset_index()["target"]):
        print("train_df and train_tsdf target are the same")
    else:
        print("train_df and train_tsdf target are different")

    # check if the train_df and train_tsdf are the same
    if train_df["timestamp"].equals(
        train_tsdf.to_data_frame().reset_index()["timestamp"]
    ):
        print("train_df and train_tsdf timestamp are the same")
    else:
        print("train_df and train_tsdf timestamp are different")

    # check if the train_df and train_tsdf are the same
    if train_df["item_id"].equals(train_tsdf.to_data_frame().reset_index()["item_id"]):
        print("train_df and train_tsdf item_id are the same")
    else:
        print("train_df and train_tsdf item_id are different")


def test_running_index_feature(train_df, train_tsdf):
    # for each item_id, get the running index from the train_tsdf and test it against train_df
    # get the running index from the train_tsdf
    train_tsdf_running_index = train_tsdf.to_data_frame().reset_index()["running_index"]
    train_df_running_index = train_df["running_index"]
    # test if the running index is the same
    if train_tsdf_running_index.equals(train_df_running_index):
        print("train_tsdf_running_index and train_df_running_index are the same")
    else:
        print("train_tsdf_running_index and train_df_running_index are different")


def test_feature_transformer(train_df, train_tsdf_original):
    # Step 1: Convert train_tsdf_original to DataFrame
    df_original = train_tsdf_original.to_data_frame().reset_index()

    # Step 2: Set index for both DataFrames
    df1 = train_df.set_index(["item_id", "timestamp"]).sort_index()
    df2 = df_original.set_index(["item_id", "timestamp"]).sort_index()

    # Step 3: Find common columns
    common_cols = [col for col in df1.columns if col in df2.columns]

    all_match = True

    for col in common_cols:
        for item_id in df1.index.get_level_values("item_id").unique():
            s1 = df1.loc[item_id][col]
            s2 = df2.loc[item_id][col]
            if not s1.equals(s2):
                print(f"Difference found in column '{col}' for item_id '{item_id}':")
                diff = s1.compare(s2)
                print(diff)
                all_match = False
            # Assert for strict checking
            assert s1.equals(s2), f"Column '{col}' mismatch for item_id '{item_id}'"

    if all_match:
        print(
            "All common columns match between train_df and train_tsdf_original for all item_ids."
        )
    else:
        print("Some columns do not match. See above for details.")
