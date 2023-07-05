Deep Learning has become the main driver of many new applications and it’s time to really look at why this is the case. With so many other options that we’ve been using for so long, why Deep Learning?

Deep Learning is popular right now because it’s easy and it works. OK you maybe thinking deep learning is easy and works, but what is it? I’m glad you asked let’s jump right to it. 😄

> Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Similarly to how we learn from experience, the deep learning algorithm would perform a task repeatedly, each time tweaking it a little to improve the outcome.
> 
> We refer to ‘deep learning’ because the neural networks have various (deep) layers that enable learning. Just about any problem that requires “thought” to figure out is a problem deep learning can learn to solve.

Deep learning differs from traditional machine learning techniques in that they can automatically learn representations from data such as images, video or text, without introducing hand-coded rules or human domain knowledge. Their highly flexible architectures can learn directly from raw data and can increase their predictive accuracy when provided with more data. For example in facial recognition, how pixels in an image create lines and shapes, how those lines and shapes create facial features and how these facial features are arranged into a face.

> _Deep learning changes how you think about representing the problem that you’re solving with analytics. It moves from telling the computer how to solve a problem to training the computer to solve the problem itself_

For example, a deep learning model known as a convolutional neural network can be trained using large numbers (as in millions) of images, such as those containing cats. This type of neural network typically learns from the pixels contained in the images it acquires. It can classify groups of pixels that are representative of a cat’s features, with groups of features such as claws, ears, and eyes indicating the presence of a cat in an image.

Deep learning is fundamentally different from conventional machine learning. In this example, a domain expert would need to spend considerable time engineering a conventional machine learning system to detect the features that represent a cat. With deep learning, all that is needed is to supply the system with a very large number of cat images, and the system can autonomously learn the features that represent a cat.

![](https://miro.medium.com/v2/resize:fit:750/0*voLrMEGfv4CMIP8C)

Credit For The Image Goes To: TechTarget

The amount of data we generate every day is staggering — currently estimated at 2.6 quintillion bytes and it’s the resource that makes deep learning possible. Since deep-learning algorithms require a ton of data to learn from, this increase in data creation is one reason that deep learning capabilities have grown in recent years. In addition to more data creation, deep learning algorithms benefit from the stronger computing power that’s available today.

Deep learning has been also instrumental in the discovery of exoplanets and novel drugs and the detection of diseases and subatomic particles. It is fundamentally augmenting our understanding of biology, including genomics, proteomics, metabolomics, the immunome, and more.

# What kind of problems does deep learning solve?

As mentioned, deep neural networks excel at making predictions based on largely unstructured data. That means they deliver best in class performance in areas such as speech and image recognition, where they work with messy data such as recorded speech and photographs.

# Should you use always deep learning instead of machine learning?

No, because deep learning can be very expensive from a computational point of view. For non-trivial tasks, training a deep-neural network will often require processing large amounts of data using clusters of high-end GPUs for many, many hours.

Given top-of-the-range GPUs can cost thousands of dollars to buy, or up to $5 per hour to rent in the cloud, it’s unwise to jump straight to deep learning.

If the problem can be solved using a simpler machine-learning algorithm such as Bayesian inference or linear regression, one that doesn’t require the system to grapple with a complex combination of hierarchical features in the data, then these far less computational demanding options will be the better choice.

Deep learning may also not be the best choice for making a prediction based on data. For example, if the dataset is small then sometimes simple linear machine-learning models may yield more accurate results — although some machine-learning specialists argue a properly trained deep-learning neural network can still perform well with small amounts of data.

# What are the drawbacks of deep learning?

One of the big drawbacks is the amount of data they require to train, with Facebook recently announcing it had used one billion images to achieve record-breaking performance by an image-recognition system. When the datasets are this large, training systems also require access to vast amounts of distributed computing power. This is another issue of deep learning, the cost of training. Due to the size of datasets and number of training cycles that have to be run, training often requires access to high-powered and expensive computer hardware, typically high-end GPUs or GPU arrays. Whether you’re building your own system or renting hardware from a cloud platform, neither option is likely to be cheap.

Deep-neural networks are also difficult to train, due to what is called the vanishing gradient problem, which can worsen the more layers there are in a neural network. As more layers are added the vanishing gradient problem can result in it taking an unfeasibly long time to train a neural network to a good level of accuracy.

# Why are deep neural networks hard to train?

As mentioned deep neural networks are hard to train because of the number of layers in the neural network. The number of layers and links between neurons in the network is such that it can become difficult to calculate the adjustments that need to be made at each step in the training process — a problem referred to as the vanishing gradient problem.

![](https://miro.medium.com/v2/resize:fit:625/0*pfV4tlsyGkOIc457)

Credit For The Image Goes To: [makeameme.org](http://makeameme.org/)

Another big issue is the vast quantities of data that are necessary to train deep learning neural networks, with training corpuses often measuring petabytes in size.

# How much does it cost to invest in deep learning?

It depends on your approach, but it will typically cost you hundreds of dollars upwards, depending on the complexity of the machine-learning task and your chosen method.

# What deep learning techniques exist?

There are various types of deep neural network, with structures suited to different types of tasks. For example, Convolutional Neural Networks (CNNs) are typically used for computer vision tasks, while Recurrent Neural Networks (RNNs) are commonly used for processing language. Each has its own specializations, in CNNs the initial layers are specialized for extracting distinct features from the image, which are then fed into a more conventional neural network to allow the image to be classified. Meanwhile, RNNs differ from a traditional feed-forward neural network in that they don’t just feed data from one neural layer to the next but also have built-in feedback loops, where data output from one layer is passed back to the layer preceding it — lending the network a form of memory. There is a more specialized form of RNN that includes what is called a memory cell and that is tailored to processing data with lags between inputs.

The most basic type of neural network is a multi-layer perceptron network, the type discussed above in the handwritten figures example, where data is fed forward between layers of neurons. Each neuron will typically transform the values they are fed using an activation function, which changes those values into a form that, at the end of the training cycle, will allow the network to calculate how far off it is from making an accurate prediction.

There are a large number of different types of deep neural networks. No one network is inherently better than the other, they just are better suited to learning particular types of tasks.

More recently, generative adversarial networks (GANS) are extending what is possible using neural networks. In this architecture two neural networks do battle, the generator network tries to create convincing “fake” data and the discriminator attempts to tell the difference between fake and real data. With each training cycle, the generator gets better at producing fake data and the discriminator gains a sharper eye for spotting those fakes. By pitting the two networks against each other during training, both can achieve better performance. GANs have been used to carry out some remarkable tasks, such as turning these dashcam videos from day to night or from winter to summer, as shown in the video below, and have applications ranging from turning low-resolution photos into high-resolution alternatives and generating images from written text. GANs have their own limitations, however, that can make them challenging to work with, although these are being tackled by developing more robust GAN variants.

Which deep learning software frameworks are available?

There are a wide range of deep learning software frameworks, which allow users to design, train and validate deep neural networks, using a range of different programming languages.

A popular choice is Google’s TensorFlow software library, which allows users to write in Python, Java, C++, and Swift, and that can be used for a wide range of deep learning tasks such as image and speech recognition, and which executes on a wide range of CPUs, GPUs, and other processors. It is well-documented, and has many tutorials and implemented models that are available.

Another popular choice, especially for beginners, is PyTorch, a framework that offers the imperative programming model familiar to developers and allows developers to use standard Python statements. It works with deep neural networks ranging from CNNs to RNNs and runs efficiently on GPUs.

Among the wide range of other options are Microsoft’s Cognitive Toolkit, MATLAB, MXNet, Chainer, and Keras.

# Deep Learning Use Case Examples

# Robotics

Many of the recent developments in robotics have been driven by advances in AI and deep learning. For example, AI enables robots to sense and respond to their environment. This capability increases the range of functions they can perform, from navigating their way around warehouse floors to sorting and handling objects that are uneven, fragile, or jumbled together. Something as simple as picking up a strawberry is an easy task for humans, but it has been remarkably difficult for robots to perform. As AI progresses, that progress will enhance the capabilities of robots.

Developments in AI mean we can expect the robots of the future to increasingly be used as human assistants. They will not only be used to understand and answer questions, as some are used today. They will also be able to act on voice commands and gestures, even anticipate a worker’s next move. Today, collaborative robots already work alongside humans, with humans and robots each performing separate tasks that are best suited to their strengths.

# Agriculture

AI has the potential to revolutionize farming. Today, deep learning enables farmers to deploy equipment that can see and differentiate between crop plants and weeds. This capability allows weeding machines to selectively spray herbicides on weeds and leave other plants untouched. Farming machines that use deep learning–enabled computer vision can even optimize individual plants in a field by selectively spraying herbicides, fertilizers, fungicides, insecticides, and biologicals. In addition to reducing herbicide use and improving farm output, deep learning can be further extended to other farming operations such as applying fertilizer, performing irrigation, and harvesting.

# Medical Imaging and Healthcare

Deep learning has been particularly effective in medical imaging, due to the availability of high-quality data and the ability of convolutional neural networks to classify images. For example, deep learning can be as effective as a dermatologist in classifying skin cancers, if not more so. Several vendors have already received FDA approval for deep learning algorithms for diagnostic purposes, including image analysis for oncology and retina diseases. Deep learning is also making significant inroads into improving healthcare quality by predicting medical events from

# Virtual assistants

Whether it’s Alexa or Siri or Cortana, the virtual assistants of online service providers use deep learning to help understand your speech and the language humans use when they interact with them.

# Translations

In a similar way, deep learning algorithms can automatically translate between languages. This can be powerful for travelers, business people and those in government.

By the way check out this funny fail by Google translate (For all you triggered Bieber fans out there right now it’s not me it’s Google’s fault OK?! :] )

![](https://miro.medium.com/v2/resize:fit:753/0*rzFFI-5G4JLbF_jI)

# PVision for driverless delivery trucks, drones and autonomous cars

The way an autonomous vehicle understands the realities of the road and how to respond to them whether it’s a stop sign, a ball in the street or another vehicle is through deep learning algorithms. The more data the algorithms receive, the better they are able to act human-like in their information processing — knowing a stop sign covered with snow is still a stop sign.

# Chatbots and service bots

Chatbots and service bots that provide customer service for a lot of companies are able to respond in an intelligent and helpful way to an increasing amount of auditory and text questions thanks to deep learning.

# Image colorization

Transforming black-and-white images into color was formerly a task done meticulously by human hand. Today, deep learning algorithms are able to use the context and objects in the images to color them to basically recreate the black-and-white image in color. The results are impressive and accurate.

# Facial recognition

![](https://miro.medium.com/v2/resize:fit:875/0*eo8t78-G9fjPPnxb)

Credit For The Image Goes To: Yann LeCun

Deep learning is being used for facial recognition not only for security purposes but for tagging people on Facebook posts and we might be able to pay for items in a store just by using our faces in the near future. The challenges for deep-learning algorithms for facial recognition is knowing it’s the same person even when they have changed hairstyles, grown or shaved off a beard or if the image taken is poor due to bad lighting or an obstruction.

# The Future of Deep Learning

Today, there are various neural network architectures optimized for certain types of inputs and tasks. Convolution neural networks are very good at classifying images. Another form of deep learning architecture uses recurrent neural networks to process sequential data. Both convolution and recurrent neural network models perform what is known as supervised learning, which means they need to be supplied with large amounts of data to learn. In the future, more sophisticated types of AI will use unsupervised learning. A significant amount of research is being devoted to unsupervised and semisupervised learning technology.

Reinforcement learning is a slightly different paradigm to deep learning in which an agent learns by trial and error in a simulated environment solely from rewards and punishments. Deep learning extensions into this domain are referred to as deep reinforcement learning (DRL). There has been considerable progress in this field, as demonstrated by DRL programs beating humans in the ancient game of GO.

Designing neural network architectures to solve problems is incredibly hard, made even more complex with many hyperparameters to tune and many loss functions to choose from to optimize. There has been a lot of research activity to learn good neural network architectures autonomously. Learning to learn, also known as metalearning or AutoML, is making steady progress.

Current artificial neural networks were based on 1950s understanding of how human brains process information. Neuroscience has made considerable progress since then, and deep learning architectures have become so sophisticated that they seem to exhibit structures such as grid cells, which are present in biological neural brains used for navigation. Both neuroscience and deep learning can benefit each other from cross-pollination of ideas, and it’s highly likely that these fields will begin to merge at some point.