### VCP
Vision classification para la clase de matamatica
***

# Judging a book by its cover
The age-old saying "don't judge a book by its cover" is often told by teachers to 
their uninterested students who dislike reading. However, seldom is their actually 
applicable in the realworld. Publishers dedicate a lot of resources for to make their 
works visually appealing and customers rightfully jude those covers to make an educated
estimate regarding the book's genre, story, them, and other literary aspects. 

That posses a question: **How much information can a cover convey?** 

The initial question was answered in the 2017 paper of similar name "Judging a Book by its Cover"
by German Research Center for Artificial Intelligence. In the process they employ the use of 
convention vision classification techniques such as CNN to estimate the genre of book by its
cover art. The research revels that CNNs have been able to understand patterns and rules set by 
publishers to define a genre. However, they leave the broder question, how much information does
the cover posses, to the readers.

In light of the various advancements in technology over the years the goal of this work is to reattempt 
the original experiment with newer technology, a tougher task, and a smaller dataset. 
***

## Dataset
The dataset used for this research comprises 3000 unique cover images of eastern light novels, and their respected
genres. It should be noted that the survivorship bias in this dataset is intentional. As unlike the original research
instead of classifying genre into sports or medicine, the goal is to classify the literary work into mystery, adventure,
comedy, etc.

The dataset will be publicly available, and readers are encouraged to give it their own attempt at answering the question.
***

## Architecture
The architecture for this experiment comprises two district branches, a **Variational Autoencoder** (VAE),
and a **Vision Transformer** (ViT). Images are often of high dimensionality are computationally expensive to 
process. In the recent decade there have been various proposals to reduce dimensionality ranging from resizing
the image to dividing the images into patches and processing the patches individually. 