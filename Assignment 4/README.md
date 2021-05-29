**TEAM MEMBERS**

Darshan jani ,
Rushiraj parmer,
Jaiveer ,
Saurabh Jain


#### Task - Create an Image Classification Network with less than 20K Params but achieves more than 99.4% Val accuracy

***SOLUTION:***

Architecture - Modified Resnet - ***MNISTResnet***
![model_architecture](https://user-images.githubusercontent.com/35656144/120026836-819e2200-bfa7-11eb-9c75-f3dd2316808e.png)


We implemented the concept of Resnet Architecture but with many significant changes like decreasing the number of channels per layer as well as the blocks

The architecture has 19K params and consists of 8 Conv layers. Due to the concept of Residual addition, we achieve 98.93% **VAL ACCURACY** with 40 epochs. 



How our Implementation differs from the Original Resnet:

- No. of input channels=1 (Obviously as the MNIST dataset has single channel images)
- The initial convolutional layer has smaller filter size, lower stride and less padding, and is not followed by a pooling layer. It also only has 3 layers, instead of 4, and has its own type of block
- In the "CIFARBasicBlock" we use a Downsampling connection with "zero padding" and all shortcuts are parameter free.
- For the function, we first slice the input with `x[:, :, ::2, ::2]`. This removes every other row and column in the image - downsampling it by simply throwing away pixels - whilst keeping the number of channels (depth) the same.
- We then double the number of channels with zeros using `pad`, which adds half the padding on to the front of the depth dimension, and half to the back.

Additional Points:

- We use Learning Rate Finder to find a suitable learning rate by plotting the graph of learning rate vs Loss. We start from an initial small learning rate of 1e-7 and go uptil a value of 10![WhatsApp Image 2021-05-28 at 9 46 57 AM](https://user-images.githubusercontent.com/35656144/120026546-148a8c80-bfa7-11eb-8719-0d3f32b4e032.jpeg)
- Training / Test Accuray with 19k params model![WhatsApp Image 2021-05-28 at 9 48 36 AM](https://user-images.githubusercontent.com/35656144/120026765-6b906180-bfa7-11eb-9a0b-7c44ef2665f7.jpeg)




      class LRFinder:
       def__init__(self, model, optimizer, criterion, device):
        
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        
        torch.save(model.state_dict(), 'init_params.pt')

      def range_test(self, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()
            
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))
                    
        return lrs, losses

      def _train_batch(self, iterator):
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred, _ = self.model(x)
                
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

      class ExponentialLR(_LRScheduler):
       def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

      def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

      class IteratorWrapper:
       def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

      def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

      def get_batch(self):
        return next(self)
        
        
