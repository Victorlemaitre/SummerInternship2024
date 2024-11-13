"""Module routines for training an expansion model"""
import argparse
from typing import Sequence, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import transpose, stack
from tensorflow.keras.layers import Layer, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Add,Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers,Model
from sklearn.utils import shuffle


from aizynthtrain.utils.configs import (
    ExpansionModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.keras_utils import (
    InMemorySequence,
    setup_callbacks,
    train_keras_model,
    top10_acc,
    top50_acc,
)


class BaseAttention(Layer):
    def __init__(self,*,dropout,num_word,k,**kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(dropout=dropout ,**kwargs)
        self.add = Add()
        self.proj = tf.Variable(initial_value=tf.random.normal(shape=(k,num_word),mean=0.0,stddev=0.2),trainable=True)


class LinAttention(BaseAttention):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self,x):
        x_proj = tf.matmul(self.proj,x)
        attn_output = self.mha(
            query = x,
            value = x_proj,
            key = x_proj
        )
        x = self.add([x,attn_output])
        return x

class FeedForward(Layer):

    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.seq = Sequential([
            Dense(d_ff,activation="gelu"),
            Dropout(dropout),
            Dense(d_model)
        ])
        self.add = Add()

    @tf.function
    def call(self,x):
        x = self.add([x,self.seq(x)])
        return x


class Block(Layer):
    def __init__(self,num_heads,head_size,d_model,d_ff,dropout,num_word,k):
        super().__init__()
        self.ln1 = LayerNormalization()
        self.at = LinAttention(num_heads=num_heads,key_dim=head_size,dropout=dropout,num_word=num_word,k=k)
        self.ln2 = LayerNormalization()
        self.ff = FeedForward(d_model,d_ff,dropout)

    @tf.function
    def call(self,x):
        x = self.at(self.ln1(x))
        x = self.ff(self.ln2(x))
        return x

class Stack(Layer):

    def __init__(self,n_blocks,num_heads,head_size,d_model,d_ff,dropout,num_word,k):
        super().__init__()
        self.seq = Sequential()

        for i in range(n_blocks):
            self.seq.add(Block(num_heads=num_heads,head_size=head_size,d_model=d_model,d_ff=d_ff,dropout=dropout,num_word=num_word,k=k))

    @tf.function
    def call(self,x):
        return self.seq(x)

class Learned_Embedding(Layer):

    def __init__(self,d_model,d_embd,input_dim,num_word):
        super().__init__()
        self.seq = Sequential([
            Dense(d_embd,activation="gelu",input_shape=(input_dim,)),
            Dense(d_model*num_word)
        ])
        self.d_model = d_model
        self.num_word = num_word

    @tf.function
    def call(self,x):
        x = tf.reshape(self.seq(x),[-1,self.num_word,self.d_model])
        return x
class Transformer(Model):
    def __init__(self,*,n_blocks,num_heads,head_size,d_model,d_ff,d_embd,input_dim,output_dim,batch_size,num_word,k,dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.Stack = Stack(n_blocks,num_heads,head_size,d_model,d_ff,dropout,num_word,k)
        self.Final = Dense(output_dim,activation="softmax")
        self.embedding_conv = Learned_Embedding(d_model=d_model, d_embd = d_embd, num_word=num_word, input_dim=input_dim)
        self.pooling = GlobalAveragePooling1D()
        self.ln = LayerNormalization()

    @tf.function
    def call(self,x):

        x = self.embedding_conv(x)
        x = self.Stack(x)
        x = self.ln(x)
        x = self.pooling(x)
        x = self.Final(x)



        return x


class ExpansionModelSequence(InMemorySequence):
    """
    Custom sequence class to keep sparse, pre-computed matrices in memory.
    Batches are created dynamically by slicing the in-memory arrays
    The data will be shuffled on each epoch end

    :ivar output_dim: the output size (number of templates)
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        super().__init__(input_filename, output_filename, batch_size)
        self.output_dim = self.label_matrix.shape[1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        idx_ = self._make_slice(idx)
        return self.input_matrix[idx_].toarray(), self.label_matrix[idx_].toarray()

    def on_epoch_end(self) -> None:
        self.input_matrix, self.label_matrix = shuffle(
            self.input_matrix, self.label_matrix, random_state=0
        )



def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line tool for training the model"""
    parser = argparse.ArgumentParser("Tool to training an expansion network policy")
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelPipelineConfig = load_config(
        args.config, "expansion_model_pipeline"
    )

    train_seq = ExpansionModelSequence(
        config.filename("model_inputs", "training"),
        config.filename("model_labels", "training"),
        config.model_hyperparams.batch_size,
    )
    valid_seq = ExpansionModelSequence(
        config.filename("model_inputs", "validation"),
        config.filename("model_labels", "validation"),
        config.model_hyperparams.batch_size,
    )



    model = Transformer(n_blocks=1, num_heads=2, head_size=64, d_model=128, d_ff=512,num_word = 64, k= 8,d_embd=1024, input_dim=train_seq.input_dim,output_dim=train_seq.output_dim,batch_size=config.model_hyperparams.batch_size)
    #get a summary of the model architecture
    input_test = tf.ones((config.model_hyperparams.batch_size,train_seq.input_dim))
    model(input_test)
    model.summary()
    

    """
    Original model architecture : 
    
    model = Sequential()
    model.add(
        Dense(
            config.model_hyperparams.hidden_nodes,
            input_shape=(train_seq.input_dim,),
            activation="elu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dropout(config.model_hyperparams.dropout))

    model.add(Dense(train_seq.output_dim, activation="softmax"))

    model.build(input_shape=(config.model_hyperparams.batch_size, train_seq.input_dim))
    model.summary()
    """

    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    train_keras_model(
        model,
        train_seq,
        valid_seq,
        "categorical_crossentropy",
        ["accuracy", "top_k_categorical_accuracy", top10_acc, top50_acc],
        callbacks,
        config.model_hyperparams.epochs,
    )




if __name__ == "__main__":
    main()
