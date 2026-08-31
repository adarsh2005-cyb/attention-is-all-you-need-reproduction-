# attention-is-all-you-need-reproduction-
Here we are reproducing the results of attention is all you need . the attention mechanism solves the problem of memory constraint which limits batching in longer sequence lenghts , it allows parallelization which can reduce computational cost by a huge diffrence


<img width="878" height="847" alt="image" src="https://github.com/user-attachments/assets/47dcba79-781d-4486-80d1-31be1b3077e9" /> 

The transformer follows this structure where an input array (x,x2,.....) is mapped to an continous representations (z,z2,.....) by the encoder which is used to generate output symbols (y,y1,.....) one at a time by the decoder
# encoder 
encoder consists of N = 6 identical layers each layer consists of two sub layers the first one is multi head self attention layer which is followed by fully - connected feed forward layer with residual coonection between each sub layer followed by an layer normalization so the output by encoder is layernorm(x + sublayer(x)) and output dimensions are Dmodel = 512
# decoder 
The structure of decoder is same as encoder where N=6 we have a feed forward network a attention mechanism residual networks and layer normalization but we modify the attention head to avoid it from attending to subsequent positions so that the the output is offset by 1 to make sure that predictions from position i are based on outputs less that position i
# positional encoding
since our model has no recurrence no convulation we add a postional encoding at the bottom of our encoder and decoder our model has same dimensions Dmodel as the embeddings 
<img width="626" height="152" alt="image" src="https://github.com/user-attachments/assets/eab87b14-0902-4986-a739-7dc293a293ee" />
we have many choices for positional encodings but we will be using sine and cosine fuctions as they allow to easily learn to attend by relative positions 
