import tensorflow as tf
import math
from tensorflow.keras import layers

def create_model_inputs(sequence_length:int) -> dict:
    """
        The create_model_inputs is used to create the input layers for hotel recommendation

        Paramters:
        sequence_length (int): The integer of window size used in sequencital preparation step

        Returns:
        Dictionary of input features
    """

    return {
        'user_id': layers.Input(name = 'user_id', shape = (1,), dtype = "string"),
        'gender': layers.Input(name = 'gender', shape = (1,), dtype = "string"),
        'age_group': layers.Input(name = 'age_group', shape = (1,), dtype = "string"),
        'sequence_hotel_ids': layers.Input(name = 'sequence_hotel_ids', shape = (sequence_length - 1,), dtype = "string"),
        'main_action_sequence': layers.Input(name = 'main_action_sequence', shape = (sequence_length - 1,), dtype = tf.float32),
        'target_hotel_id': layers.Input(name = 'target_hotel_id', shape = (1,), dtype = "string"),
    }

def encode_input_features(
    inputs:dict,
    dict_to_include_feature:dict,
    sequence_length:int,
    df_hotel_style,
    dict_features:dict
):
    include_user_id = dict_to_include_feature['include_user_id']
    include_user_features = dict_to_include_feature['include_user_features']
    include_hotel_features = dict_to_include_feature['include_hotel_features']

    encoded_other_features = []
    encoded_transformer_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")
    if include_user_features:
        # USER_FEATURES = 'gender' and 'age_group'
        other_feature_names.extend(dict_features['USER_FEATURES'])

    ## Encode user features
    for feature_name in other_feature_names:
        # Convert the string input values into integer indices.
        vocabulary = dict_features['CATEGORICAL_FEATURES_WITH_VOCABULARY'][feature_name]
        idx = layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a hotel embedding encoder
    hotel_vocabulary = dict_features['CATEGORICAL_FEATURES_WITH_VOCABULARY']["hotel_id"]
    hotel_embedding_dims = int(math.sqrt(len(hotel_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    hotel_index_lookup = layers.StringLookup(
        vocabulary=hotel_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="hotel_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    hotel_embedding_encoder = layers.Embedding(
        input_dim=len(hotel_vocabulary),
        output_dim=hotel_embedding_dims,
        name=f"hotel_embedding",
    )
    # Create a vector lookup for hotel_style.
    hotel_style_vectors = df_hotel_style.to_numpy()
    hotel_style_lookup = layers.Embedding(
        input_dim=hotel_style_vectors.shape[0],
        output_dim=hotel_style_vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(hotel_style_vectors),
        trainable=False,
        name="hotel_style_vector",
    )
    # Create a processing layer for hotel_style.
    hotel_embedding_processor = layers.Dense(
        units=hotel_embedding_dims,
        activation="relu",
        name="process_hotel_embedding_with_style_name",
    )

    ## Define a function to encode a given hotel id.
    def encode_hotel(hotel_id):
        # Convert the string input values into integer indices.
        hotel_idx = hotel_index_lookup(hotel_id)
        hotel_embedding = hotel_embedding_encoder(hotel_idx)
        encoded_hotel = hotel_embedding
        if include_hotel_features:
            hotel_style_vector = hotel_style_lookup(hotel_idx)
            encoded_hotel = hotel_embedding_processor(
                layers.concatenate([hotel_embedding, hotel_style_vector])
            )
        return encoded_hotel

    ## Encoding target_hotel_id
    target_hotel_id = inputs["target_hotel_id"]
    encoded_target_hotel = encode_hotel(target_hotel_id)

    ## Encoding sequence hotel_ids.
    sequence_hotels_ids = inputs["sequence_hotel_ids"]
    encoded_sequence_hotels = encode_hotel(sequence_hotels_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=hotel_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Your existing code to get sequence_purchases:
    sequence_purchases = inputs["main_action_sequence"]
    sequence_purchases_shape = tf.shape(sequence_purchases)  # Get the shape of sequence_purchases.
    sequence_purchases = tf.ones_like(sequence_purchases, dtype=sequence_purchases.dtype)  # Create a tensor of ones with the same shape and dtype.

    # Alternatively, if you need to expand dimensions, do it after creating the tensor of ones:
    sequence_purchases = tf.expand_dims(sequence_purchases, -1)

    # Your subsequent code where you used sequence_purchases:
    encoded_sequence_hotels_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_hotels + encodded_positions), sequence_purchases]
    )

    # Construct the transformer inputs.
    for i in range(sequence_length - 1):
        feature = encoded_sequence_hotels_with_poistion_and_rating[:, i, ...]
        feature = tf.expand_dims(feature, 1)
        encoded_transformer_features.append(feature)
    encoded_transformer_features.append(encoded_target_hotel)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features


def create_model(dict_to_include_feature:dict, 
                 hidden_units:list, 
                 sequence_length:int, 
                 dropout_rate:float, 
                 df_hotel_style, 
                 dict_features:dict,
                 output_units:int):
    """
        The create_model is used to build the behavioral sequential transformer model (BST model).

        Parameters:
        dict_to_include_feature (dict): Contain a boolean value for include_user_id, include_user_features, and include_hotel_features
        hidden_units (list): Specify number of hidden layers and number of units
        sequence_length (int): Specify total number the sequence length -> usually will minus 1 to after exclude 1 value for a feature to predict -> use to determine number of head in transformers
        dropout_rate (int) : Ratio of drop out in MLP/FC layers
        df_hotel_style (pd.DataFrame): A dataframe contain the multi-hot encoding value of the hotel style
        dict_features (dict): A dictionary contain the available feature and the vocaburary of the available feature

        Return:
        Keras bst model
    """

    inputs = create_model_inputs(sequence_length)
    transformer_features, other_features = encode_input_features(
        inputs, dict_to_include_feature, 
        sequence_length, df_hotel_style, dict_features
    )

    # Create a multi-headed attention layer.
    num_heads = sequence_length - 1

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=output_units), activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model