class AttentionLayer(Layer):
    
    def __init__(self, output_dim=64, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                shape=(1, 1) + (self.output_dim, self.output_dim),
                                initializer='glorot_uniform',
                                trainable=True)
        self.W2 = self.add_weight(name='W2',
                                shape=(1, 1) + (self.output_dim, self.output_dim),
                                initializer='glorot_uniform',
                                trainable=True)
        self.W3 = self.add_weight(name='W3',
                                shape=(1, 1) + (self.output_dim, self.output_dim),
                                initializer='glorot_uniform',
                                trainable=True)

        super(AttentionLayer, self).build(input_shape) 


    def call(self, x):        

        f = K.conv2d(x,kernel=self.W1,strides=(1, 1), padding='same') 
        g = K.conv2d(x,kernel=self.W2,strides=(1, 1), padding='same')  
        h = K.conv2d(x,kernel=self.W3,strides=(1, 1), padding='same')  

        batchdot1 = K.batch_dot(K.reshape(g, shape=[K.shape(g)[0], K.shape(g)[1]*K.shape(g)[2], K.shape(g)[3]]), 
                        K.permute_dimensions(K.reshape(f, shape=[K.shape(f)[0], K.shape(f)[1]*K.shape(f)[2], K.shape(f)[3]]), (0, 2, 1)))  

        softmaxactivate1 = K.softmax(batchdot1)  

        batchdot2 = K.batch_dot(softmaxactivate1, 
                        K.reshape(h, shape=[K.shape(h)[0], K.shape(h)[1]*K.shape(h)[2], K.shape(h)[3]]))  

        reshape1 = K.reshape(batchdot2, shape=K.shape(x)) 
        
        sum_x = reshape1 + x

        return sum_x        


    def compute_output_shape(self, input_shape):
        return input_shape
