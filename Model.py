import torch 
import torch.nn as nn 
import pandas as pd
import numpy as np
import torch.optim as optim
import math
import random



class Transformer(nn.Module) : 

    def __init__(self, data, char_idx , embedding_size, attention_head_size, linear_size, dropout = 0.2):
        super().__init__()

        self.sow, self.eow = '$', '£'
        self.pad = '%'

        # our indexed vocab
        self.char_idx = char_idx

        self.data = data 
        self.embedding_size = embedding_size
        self.att_head_size = attention_head_size
        self.linear_size = linear_size
        self.max_length = len(self.data[0][0])

        # a softmax function
        self.soft = nn.Softmax(dim = -1)

        # our embedding layer :
        self.embedding = nn.Embedding(num_embeddings=len(self.char_idx), embedding_dim=self.embedding_size, padding_idx=self.char_idx[self.pad])

        # for the encoder network : 
        self.encoder_component_linear = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.encoder_reshape_linear = nn.Linear(self.embedding, self.embedding)
        self.encoder_seq = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size * 2), 
                                         nn.ReLU(), 
                                         nn.Dropout(p = dropout), 
                                         nn.Linear(self.embedding_size * 2, self.embedding_size)
                                         )
        
        # we can create a fixed size position encoding and truncate it depending on the size of the input
        self.pos_embedding = self.positional_encoding()


        # for the decoder network :
        self.decoder_linear = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.mask = torch.triu(torch.full((self.max_length, self.max_length), fill_value=float("-inf")), diagonal=1) 
        self.multi_head_decoder_linear = nn.Linear(self.embedding_size, self.embedding_size)



    def positional_encoding(self):
        """
        this method is to add a positionnal encoding for the a given embedded input sequence

        returns : 
            - embedded sequence with positionnal encoding     
        """

        # the shape of the embedded vec (batch size, sequence length, embedding size)
        pos_embedding = torch.zeros(size=(self.max_length, self.embedding_size))

        for char in range(self.max_length):
            for pos in range(self.embedding_size):
                if pos % 2 : # when the position index is odd
                    pos_embedding[char][pos] = np.cos(char / (10000 **(2 * pos / self.embedding_size)))
                else : 
                    pos_embedding[char][pos] = np.sin(char / (10000 **(2 * pos / self.embedding_size)))

        return pos_embedding
    
        
    def residual(self, initial_emb , att_emb ):
        """
        this method is for the add & norm step 
        input : 
                initial_emb : embedding before multi head attention 
                att_emb : output embedding of the multi head attention 
        """
        # we add the two input emebeddings 
        tmp_emb = initial_emb + att_emb

        # we standardize our summed embedding 
        mean = torch.mean(tmp_emb, dim = 1)
        std_2 = torch.pow(torch.std(tmp_emb, dim = 1), 2) + torch.full(mean.size(), fill_value=0.0001)     
        final_emb = (tmp_emb - mean) / np.sqrt(std_2)

        return final_emb
    
    def multi_head_attention(self,Query, Key , Value, head_dim ,mask = False):
        """
        computes the attention filter of a given input embedding 
        """

        # we reshape the input emb 
        
        prod = torch.matmul(Query, Key.transpose(-2, -1)) / math.sqrt(Query.size()[-1])
        
        if mask :
            msk = self.mask.reshape(prod.size())
            prod = prod + msk

        # our scaled attention filter
        scaled_prod = self.soft(prod)

        # apply the attention filter to the values
        scaled_prod = torch.matmul(scaled_prod, Value)

        final_out = scaled_prod.reshape(scaled_prod.size(0),scaled_prod.size(1), self.att_head_size * head_dim)
        
        final_out = self.multi_head_encoder_linear(final_out) 

        return final_out

    

    def encoder(self, batch):
        """
        we encode a batch of data
        """
        emb_batch = self.embedding(batch)

        # we add positional embeddings to the input embedding
        
        pos_emb = self.pos_embedding.reshape(emb_batch) # first we reshape the positional embedding to the size of the input batch
        pos_embedded_batch = emb_batch + pos_emb

        # now we feed our embedding to an attention mechanism do we feed the same data to all attention heads or we seperate it 
        out1 = self.encoder_component_linear(pos_embedded_batch)
        head_dim = self.embedding_size // self.att_head_size
        resh_out = out1.reshape(emb_batch.size(0), emb_batch.size(1), self.att_head_size, 3*head_dim)
        resh_out = resh_out.permute(0, 2, 1, 3)

        Query, Key , Value = resh_out.chunk(3, dim = -1) 
        attention_output = self.multi_head_attention(Query, Key, Value, mask = False)

        # we apply normalization :
        norm_out = self.residual(pos_embedded_batch,attention_output)

        # we pass the normalized attention filtered embedding through a feed forward network 
        output = self.encoder_seq(norm_out)

        # we apply normalization again : 
        output = self.residual(norm_out, output)
        
        return output # the final encoder output vector with applied attention 


    def decoder(self, encoder_output):
        """
        the decoder network of our transformer, i need to pay attention to the masking thing (for tomorrow :) )
        """

        
