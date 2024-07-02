import pandas as pd
import tensorflow as tf

# De-list
def join_value_in_sequence(df, col_list):
    for col in col_list:
        df[col] = df[col].apply(lambda x: [str(value) for value in x])
        df[col] = df[col].apply(lambda x: ",".join(x))
    return df
    
def create_sequences(values, widowing_spec):
    sequences = []
    start_index = 0

    window_size = widowing_spec['SEQUENCE_LENGTH']
    step_size = widowing_spec['STEP_SIZE']

    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences

def merge_sequece_lists(df, col_list, widowing_spec):
    for col in col_list:
        if col in df.columns:
            new_col = col+'_sequence'
            df[new_col] = df[col].apply(lambda x: create_sequences(x, widowing_spec))
    
    return df

def get_sequence_data(df_event, df_user_info, widowing_spec, num_required_seen_hotel, dict_main_action_encoder, DATA_PATH, full_loop_flag):

    # Encode main action
    df_event['main_action'] = df_event['main_action'].replace(dict_main_action_encoder)
    print('Successfully encoded')

    # Create sequence
    df_event.sort_values(by=['user_id', 'event_timestamp'], inplace=True)
    df_event = df_event.groupby('user_id').agg(list).reset_index()
    del df_event['event_timestamp']

    def count_item(row):
        return len(list(set(row)))

    df_event = df_event[df_event.hotel_id.apply(count_item) >= num_required_seen_hotel].reset_index(drop=True)

    col_list = ['main_action', 'hotel_id', 'complete_flag', 'duration'] # List to generate sequence

    df_event = merge_sequece_lists(df_event, col_list, widowing_spec)
    df_event.drop(columns=col_list, inplace=True)

    hotel_id_data = df_event[["user_id", "hotel_id_sequence"]].explode("hotel_id_sequence", ignore_index=True)
    main_action_data = df_event[["main_action_sequence"]].explode("main_action_sequence", ignore_index=True)
    complete_flag_data = df_event[["complete_flag_sequence"]].explode("complete_flag_sequence", ignore_index=True)
    duration_data = df_event[["duration_sequence"]].explode("duration_sequence", ignore_index=True)

    df_event_transformed = pd.concat([hotel_id_data, main_action_data, complete_flag_data, duration_data], axis=1, ignore_index=True)
    df_event_transformed.columns = ['user_id', 'hotel_id_sequence', 'main_action_sequence', 'complete_flag_sequence', 'duration_sequence']

    df_event_transformed = df_event_transformed[~df_event_transformed['hotel_id_sequence'].isnull()]

    df_event_transformed = join_value_in_sequence(df_event_transformed, ['hotel_id_sequence',
                                                                        'main_action_sequence',
                                                                        'complete_flag_sequence',
                                                                        'duration_sequence'])
    
    df_event_transformed = df_event_transformed.merge(df_user_info, how='left', on='user_id')

    if full_loop_flag:
        eval_size = 0.15
        test_size = 0.15

        print('Split eval size =', eval_size, 'for users =', df_event_transformed.user_id.nunique())

        n = df_event_transformed.shape[0]
        num_splits = 3 

        start_train = 0
        end_train = int((n * (1 - (test_size + eval_size))) - 1)

        start_test = end_train + 1
        end_test = start_test + int(n * test_size) - 1

        start_eval = end_test + 1
        end_eval = n - 1

        # Split the dataframe into train, test, and evaluation sets using the calculated index ranges
        df_event_transformed = df_event_transformed.sample(frac=1).reset_index(drop=True)
        df_train = df_event_transformed[start_train:end_train+1]
        df_test = df_event_transformed[start_test:end_test+1]
        df_eval = df_event_transformed[start_eval:end_eval+1]

        df_train.to_csv(f'{DATA_PATH}/train.csv', index=False, header=True, sep="|")
        df_test.to_csv(f'{DATA_PATH}/test.csv', index=False, header=True, sep="|")
        df_eval.to_csv(f'{DATA_PATH}/eval.csv', index=False, header=True, sep="|")
        output = {
            'train':df_train,
            'test':df_test,
            'eval':df_eval
        }
    else:
        df_event_transformed.to_csv(f'{DATA_PATH}/full_train.csv', index=False, header=True, sep="|")
        output = {
            'full_train':df_event_transformed
        }
    return output

def get_dataset_from_csv(csv_file_path, target_col_name, shuffle=False, batch_size=1024):

    def process(features):
        hotel_id_sequence_string = features["hotel_id_sequence"]
        sequence_hotel_ids = tf.strings.split(hotel_id_sequence_string, ",").to_tensor()

        # The last movie id in the sequence is the target movie.
        features["target_hotel_id"] = sequence_hotel_ids[:, -1]
        features["sequence_hotel_ids"] = sequence_hotel_ids[:, :-1]

        # duration_string = features["complete_flag_sequence"]
        # duration_string = features["main_action_sequence"]
        duration_string = features[target_col_name]
        sequence_duration = tf.strings.to_number(
            tf.strings.split(duration_string, ","), tf.dtypes.float32
        ).to_tensor()

        # The last rating in the sequence is the target for the model to predict.
        target = sequence_duration[:, -1]
        features[target_col_name] = sequence_duration[:, :-1]

        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        num_epochs=1,
        header=True,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset