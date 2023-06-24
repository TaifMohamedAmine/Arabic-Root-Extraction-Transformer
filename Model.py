import torch 
import torch.nn as nn 
import numpy as np
import torch.optim as optim
import math
import random




class Transformer(nn.Module) : 

    def __init__(self, data, char_idx : dict, num_epochs :int ,  batch_size : int , embedding_size : int, attention_head_size : int, linear_size : int, learning_rate = 0.001 , dropout = 0.2) :
        super().__init__()

        self.sow, self.eow = '$', '£'
        self.pad = '%'

        # how many times the decoder is repeated : 
        self.repeated = 3

        # our indexed vocab
        self.char_idx = char_idx

        self.data = data 
        self.ratio = 0.8 # training data ratio
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.att_head_size = attention_head_size
        self.linear_size = linear_size
        self.max_length = len(self.data[0][0])

        # we index our data into integer sequences : 
        self.seq_data = self.data_2_seq()

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
        self.mask = torch.triu(torch.full((self.max_length, self.max_length), fill_value=float("-inf")), diagonal=1) 
        self.decoder_component_linear = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.encoder_2_decoder_linear = nn.Linear(self.embedding_size, 2 * self.embedding_size)
        self.decoder_query_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.decoder_seq = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size * 2), 
                                         nn.ReLU(), 
                                         nn.Dropout(p = dropout), 
                                         nn.Linear(self.embedding_size * 2, self.embedding_size)
                                        )
        self.decoder_reshape_linear = nn.Linear(self.embedding_size, len(self.char_idx)) # dinal linear layer of the decoder

        # we define our optimizer : 
        self.opt = optim.AdamW(self.parameters(), lr = learning_rate)

        # we define our loss function : 
        self.loss = nn.CrossEntropyLoss(ignore_index=self.char_idx[self.pad])


    def data_2_seq(self): 
        '''
        this function indexes our text data
        '''
        final_data = []
        for instance in self.data : 
            try : 
                word = [self.char_idx[char] for char in instance[0]]
                root = [self.char_idx[char] for char in instance[1]]
            except : 
                raise Exception("vocab doesnt contain a character from the word/root : ", instance)
            final_data.append([word, root])
        return final_data
    
    
    def prepare_data(self):
        """
        prepare our data for training/validation
        """
        data_size = len(self.data)
        shuffled_data = random.sample(self.seq_data, data_size)
        middle_index = int(data_size * self.ratio)
        train_data, val_data = shuffled_data[:middle_index], shuffled_data[middle_index:]
        train_batches = [[train_data[i:i + self.batch_size][0],train_data[i:i + self.batch_size][1]] for i in range(0, data_size, self.batch_size)]
        val_batches = [val_data[i:i + self.batch_size] for i in range(0, data_size, self.batch_size)]

        return train_batches, val_batches



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
    
    def multi_head_attention(self, Query, Key , Value, head_dim : int,mask = False) -> torch.tensor:
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
        
        final_out = self.encoder_reshape_linear(final_out) 

        return final_out

    

    def encoder_network(self, batch : list) -> torch.tensor:
        """
        we encode a batch of data
        """
        emb_batch = self.embedding(batch)

        # we add positional embeddings to the input embedding
        
        pos_emb = self.pos_embedding.reshape(emb_batch) # first we reshape the positional embedding to the size of the input batch
        pos_embedded_batch = emb_batch + pos_emb

        for _ in range(self.repeated):
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
            pos_embedded_batch = output
        
        return output # the final encoder output vector with applied attention 


    def decoder_network(self, encoder_output : torch.tensor):
        """
        the decoder network of our transformer, i need to pay attention to the masking thing (for tomorrow :) )
        """

        # we start by feeding an input starting token 
    
        word_list = [[self.char_idx[self.sow]]] * encoder_output.size(0) # shape : (1, batch_size)
        
        for i in range(5):
            emb_input = self.embedding(word_list) # shape : (batch size , curr seq length, embedding size)
            pos_embedding = self.positional_encoding[:len(word_list[0]), :] # shape : (curr seq length , embedding size)
            emb_input += pos_embedding
            
            # we repeat the decoder for a given number of times
            for _ in range(self.repeated): 
                out1 = self.decoder_component_linear(emb_input)
                head_dim = self.embedding_size // self.att_head_size
                resh_out = out1.reshape(emb_input.size(0), emb_input.size(1), self.att_head_size, 3*head_dim)
                resh_out = resh_out.permute(0, 2, 1, 3)
                Query, Key , Value = resh_out.chunk(3, dim = -1)
                out_attention = self.multi_head_attention(Query, Key, Value, head_dim , mask = True)
                res1 = self.residual(emb_input, out_attention)
            
                # we prepare for the multi head cross attention step :
                enc_out = self.encoder_2_decoder_linear(encoder_output) # shape : (batch size, max seq length, 2 * embedding dim)
                Q = self.decoder_query_linear(res1) # shape : (batch size , curr input size, embedding dim)

                # we reshape everything : 
                enc_reshaped = enc_out.reshape(emb_input.size(0),emb_input.size(1), self.att_head_size, 2 * head_dim) # new shape : (batch size, curr input size , num heads, 2 * head dim)
                Q = Q.reshape(emb_input.size(0), emb_input.size(1), self.att_head_size, head_dim) # new shape : (batch size, curr input size , num heads, head dim)
                enc_reshaped = enc_reshaped.permute(0, 2, 1, 3)
                Q = Q.permute(0, 2, 1, 3)

                K, V = enc_reshaped.chunk(2, dim=-1) 

                # we apply the cross multi head attention without a mask ofc
                attention2 = self.multi_head_attention(Q, K, V,head_dim, mask=False)

                # we add residual : 
                res2 = self.residual(res1,attention2)

                # we run the current output through the decoder feed forward network 
                fd_output = self.encoder_seq(res2)

                # we apply the final residual : 
                res_output = self.residual(res2, fd_output)
                emb_input = res_output

            # we apply the final reshape linear layer : 
            final_output = self.soft(self.decoder_reshape_linear(res_output)) # shape : (batch size, curr seq length, vocab size)
            
            # we select the most probable output characters : 
            chars = torch.argmax(final_output, dim = -1).flatten().tolist()

            # we add them as input for the next interation of the decoder
            word_list = [sublist + [element] for sublist, element in zip(word_list, chars)]
        
        return final_output ,word_list 
            

    def fit(self): 
        """
        this is a method that assembles our transformer
        """
        # we first split our data into batches : 
        train_batches, self.val_batches = self.prepare_data()

        for epoch in range(self.num_epochs):
            print(f"epoch num {epoch}")
            # we shuffle our batches : 
            train_batches = random.sample(train_batches, len(train_batches))
            for batch in train_batches :
                word_batch, root_batch = zip(*batch)
                self.opt.zero_grad()
                encoder_output = self.encoder(list(word_batch))  
                proba_output, word_list = self.decoder(encoder_output)
                # we prepare the dimension of vectors for the loss
                target_words = torch.tensor(root_batch)
                batch_size, seq_length, emb_size = proba_output.size()
                
                proba_output = proba_output.view(batch_size * seq_length, emb_size)
                target_words = target_words.view(batch_size * seq_length)

                # we finally calculate the loss 
                loss = self.loss(proba_output, target_words)
                loss.backward()
                self.opt.step() 

                print("the current loss of this batch is : ", loss.item())
            
        torch.save(self.state_dict(), 'Transformer/model_par.pt')
        