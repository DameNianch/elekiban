# executor.training
This is WIP. The design is likely to change.

## train()
Used for training a single deep learning model. Here red represents what the model learns. Also, it doesn't matter if the model is multitasking.

```mermaid
graph TD;
subgraph Common Training 
    Input-->Model:::trainable;
    Model-->Output;
    classDef trainable fill:#902
end
```

## Trainer
This may be deprecated. Used for training a single deep learning model. Here red represents what the model learns. Also, it doesn't matter if the model is multitasking.

```mermaid
graph TD;
subgraph Common Training 
    Input-->Model:::trainable;
    Model-->Output;
    classDef trainable fill:#902
end
```

## SerialTrainer

### DCGAN
If the model is complicated like GAN, it is difficult to handle it with train() or Trainer. For complex learning, I defined the flow in advance and let it learn. 

```mermaid
graph TD
subgraph Train Generator
    i0(Noise)-->g0(Generator):::trainable;
    g0-->d0(Discriminator);
    d0-->f0(False);
    classDef trainable fill:#902
end
subgraph Train Discriminator
    i2(True Image)-->d1(Generator);
    i1(Noise)-->g1(Generator);
    g1-->d1(Discriminator):::trainable;
    d1-->o1(True or False);
    classDef trainable fill:#902
end
```
There are two inputs for training the discriminator, which can be expressed by combining CustomPipe and LabelPipe and using MixedPipe as follows. After that, by repeatedly learning these two in order, it is possible to learn each model in competition. The origin of "SerialTrainer" is to arrange and repeat the learning flow in a "serial", no matter how many models are intricately intertwined.

```mermaid
graph TD
subgraph Train Generator
    i0(Noise)-->g0(Generator):::trainable;
    g0-->d0(Discriminator);
    d0-->f0(False);
    classDef trainable fill:#902
end
subgraph Train Discriminator
    i1(Input)-->g1(Generator);
    g1-->d1(Discriminator):::trainable;
    d1-->o1(True or False);
    classDef trainable fill:#902
end
```
### Pseudo Conditional GAN
If you want to fix the generated image for a certain noise (like CGAN), just adding a learning flow is enough. After that, let's tune how fast you learn these learning. Please use the sample for details.However, care should be taken that the True Label is given as noise and the Discriminator is not learned.

```mermaid
graph TD
subgraph Train Generator
    i0(Noise)-->g0(Generator):::trainable;
    g0-->d0(Discriminator);
    d0-->f0(False);
    classDef trainable fill:#902
end
subgraph Train Generator fit True
    i2(True Label)-->g2(Generator):::trainable;
    g2-->f2(True Image);
    classDef trainable fill:#902
end
subgraph Train Discriminator
    i1(Input)-->g1(Generator);
    g1-->d1(Discriminator):::trainable;
    d1-->o1(True or False);
    classDef trainable fill:#902
end
```
