from kaffe.tensorflow import Network

class FCN(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .conv(1, 1, 1, 1, 1, relu=False, name='score-dsn5')
             .deconv(32, 32, 1, 16, 16, padding='VALID', relu=False, name='upsample_16'))

        (self.feed('conv1_2')
             .conv(1, 1, 1, 1, 1, relu=False, name='score-dsn1'))

        (self.feed('conv2_2')
             .conv(1, 1, 1, 1, 1, relu=False, name='score-dsn2')
             .deconv(4, 4, 1, 2, 2, padding='VALID', relu=False, name='upsample_2'))

        (self.feed('conv3_3')
             .conv(1, 1, 1, 1, 1, relu=False, name='score-dsn3')
             .deconv(8, 8, 1, 4, 4, padding='VALID', relu=False, name='upsample_4'))

        (self.feed('conv4_3')
             .conv(1, 1, 1, 1, 1, relu=False, name='score-dsn4')
             .deconv(16, 16, 1, 8, 8, padding='VALID', relu=False, name='upsample_8'))