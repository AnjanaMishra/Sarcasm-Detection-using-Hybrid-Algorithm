def build_model(num_units=128, num_layers=2):
    # BERT Input
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    token_type_ids = Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

    # BERT Encoder
    bert_output = bert_encoder([input_ids, attention_mask, token_type_ids])
    bert_embedding = bert_output.last_hidden_state

    # Bi-LSTM Layers
    lstm_output = bert_embedding
    for _ in range(num_layers):
        lstm_output = Bidirectional(LSTM(num_units, return_sequences=True))(lstm_output)

    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_output, lstm_output)

    # Global Average Pooling
    pooled_output = tf.reduce_mean(attention_output, axis=1)

    # Dense Layers
    dense1 = Dense(128, activation='relu')(pooled_output)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    output = Dense(1, activation='sigmoid')(dropout2)

    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    return model
