# Auto-encoding Molecules: Graph-Matching Capabilities Matter
Autoencoders are effective deep learning models that can function as generative models and learn latent representations for downstream tasks. The use of graph autoencoders--—with both encoder and decoder implemented as message passing networks--—is intriguing due to their ability to generate permutation-invariant graph representations. However, this approach faces difficulties because decoding a graph structure from a single vector is challenging, and comparing input and output graphs requires an effective permutation-invariant similarity measure, which is computationally expensive. As a result, many studies rely on approximate methods. In this work, we explore the effect of graph matching precision on the training behavior and generation capabilities of a VAE.
Our contribution is two-fold: we propose a transformer-based message passing graph decoder as an alternative to a graph neural network decoder, that is more robust and expressive by leveraging global attention mechanisms effectively. We show that the precision of graph matching has significant impact on training behavior and is essential for effective de novo (molecular) graph generation.

[View the PDF](mainfig.pdf)