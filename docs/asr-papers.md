# Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks
We propose a method called Connectionist Temporal Classification (CTC) for training RNNs on tasks where real-valued unsegmented input streams are annotated with strings of discrete labels (e.g. handwriting recognition, speech recognition, gesture recognition)
Terminology: we refer to the task of labelling unsegmented data sequences as temporal classification, and independent labelling of each time-step, or frame, of the input sequence as framewise classification
Let V be a vocabulary of N classes (for ASR, classes may be letters, including a class for "space").
Let E be the extended vocabulary of N+1 classes, including "blank" class that means no prediction at the current frame (not the same as space character).
CTC network performs per-frame classification to these N+1 classes. So, given K frames, model outputs probabilities of shape (K, N+1). This allows to calculate the probability of any sequence from E^T.
We then define a many-to-one mapping B from E^T to to the set of any-length (up to length T) sequences of vocabulary V (denote as V^star).
The mapping B: E^T -> V^star works like this: we first remove all repetitions, and then remove all blank tokens (not vice versa).
Let the probability of any sequence S from V^star be the sum  probabilities of all X in E^T such that B(X) = S. The probability of X is the product of probabilities of all its tokens.
Given predicted probabilities of shape (K, N+1), during inference we want to find the most probable sequence from V^star, and during training we want to calculate the probability of the target sequence from V^star. Both tasks are intractable.
We can approximate inference with greedy ("best path") search (taking argmax and then applying B mapping), or, better, by our proposed Prefix search decoding (fig. 2). It relies on the observation that the outputs of a trained CTC network tend to form a series of spikes separated by strongly predicted blanks (fig. 1), and hence we choose boundary points where the probability of observing a blank label is above a certain threshold. We then calculate the most probable labelling for each section individually. In practice, prefix search generally outperforms greedy search.
...TODO
TODO https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html
On TIMIT dataset, CTC outperformed both a baseline HMM (hidden Markov Model) recogniser and an HMM-RNN hybrid with the same RNN architecture.
A key difference between CTC and HMM is that CTC does not explicitly segment its input sequences. Determining the segmentation is a waste of modelling. For tasks where segmentation is required it would seem problematic to use CTC (however, CTC is suitable where approximate segmentation is sufficient).
Further we intend to pursue an hierarchy of temporal classifiers, where the labellings at one level (e.g. letters) become inputs for the labellings at the next (e.g. words).


# On-the-fly lattice rescoring for real-time automatic speech recognition
We propose an algorithmic framework for rescoring lattices (a DAG of transcription hypotheses) on-the-fly.
TODO


# Sequence Transduction with Recurrent Neural Networks
We propose RNN transducer for sequence-to-sequence tasks (such as phoneme recognition).
Let we have input vectors X and output vectors Y. In our case, we assume that the output space is discrete, output vectors are one-hot vectors, and the output vocabulary is extended with additional null token (meaning "output nothing"). However the method can be readily extended to continuous output spaces. As in CTC, we assume that the location of the null symbols determines an alignment between the input and output sequences, and we refer to the sequences containing null symbols as "alignments". Given X, the RNN transducer defines a conditional distribution over all possible alighments. This distribution is then collapsed onto the distribution over output tokens by removing null tokens.
The prediction network G (sec. 2.1) is a RNN that accepts and returns one-hot vectors from the extended alphabet with null symbol (the latter is represented as zero vector on input space, and as an additional element in output space, so the input vectors have length K, and the output vectors have length K+1). The prediction RNN can be either a one-layer RNN (eq. 2-3) or LSTM (eq. 4-8). The prediction network attempts to model each element of Y given the previous ones; it is therefore similar to a standard next-step-prediction RNN, only with the added option of making null predictions.
The transcription network F (sec. 2.2) is a bidirectional RNN. Bidirectionality is preferred because each output vector depends on the whole input sequence; however we have not tested to what extent this impacts performance. For a task with K output labels, the output layer of the transcription network is size K + 1, just like the prediction network. So, the transcription network is similar to a CTC RNN.
Let the transcription network outputs T vectors (sec 2.3). Let us consider any single vector from these vectors. Also, consider any single vector from the transcription network outputs. We can add them element-wise and them perform softmax (eq. 12, 13) to yield the output distribution over output alphabet and the null token. The probability of the null token can be interpreted as the need to shift T by 1.
In fig. 1, the set of possible paths from the bottom left to the terminal node in the top right corresponds to the complete set of alignments between x and y. Therefore all possible input-output alignments are assigned a probability, the sum of which is the total probability P(y|x) of the output sequence given the input sequence. A similar lattice could be drawn for any finite y. So, we have defined a distribution over all possible output sequences, given a single input sequence. A naive calculation of P(y|x) from the lattice would be intractable. However an efficient forward-backward algorithm is described in sec. 2.4.
At test time, we employ a fixed-width beam search through the tree of output sequences.
The improved version of RNN Transducer that includes the "output network", is described in "Speech Recognition with Deep Recurrent Neural Networks". An excerpt from this paper is below: CTC defines a distribution over phoneme sequences that depends only on the acoustic input sequence x. It is therefore an acoustic-only model. A recent augmentation, known as an RNN transducer combines a CTC-like network with a separate RNN that predicts each phoneme given the previous ones, thereby yielding a jointly trained acoustic and language model (IMO the authors mean that CTC network does not model the language autoregressively, but i think it models the language in the same way as BERT does). Whereas CTC determines an output distribution at every input timestep, an RNN transducer determines a separate token distribution for every combination of input timestep t and output timestep u, that covers the K phonemes plus null token. Intuitively the network "decides" what to output depending both on where it is in the input sequence and the outputs it has already emitted. For a length U target sequence z, the complete set of TU decisions jointly determines a distribution over all possible alignments between x and z, which can then be integrated out with a forward-backward algorithm to determine log P(z|x).
For decoding see https://www.youtube.com/watch?v=dgsDIuJLoJU
IMO, the latter variant is similar to RNN encoder-decoder with attention, where the decoder-to-encoder attention is reduced to the following fixed scheme: at step t we look only at the encoder step i, and we can produce either any token without changing i, or a blank token with adding one to i.


# Speech Recognition with Deep Recurrent Neural Networks
We inversigate deep LSTMs for ASR. We also present an enhancement to a recently introduced RNN transducer: an additional "output network".
In the original formulation P(token|audio, prev_tokens) was defined by taking an "acoustic" distribution P(token|audio) from the CTC network, a "linguistic" distribution P(token|prev_tokens) from the prediction network, then multiplying the two together and renormalising.
We propose to instead feed the hidden activations of both networks into a separate feedforward output network, whose outputs are then normalised with a softmax function to yield P(token|audio, prev_tokens). This allows a richer set of possibilities for combining linguistic and acoustic information and appears to lead to better generalisation: the number of deletion errors encountered during decoding is reduced.
RNN transducers appear to work better when initialised with the weights of a pretrained CTC network and a pretrained next-step prediction network.
In this work we pretrain the prediction network on the phonetic transcriptions of the audio training data; however for large-scale applications it would make more sense to pretrain on a separate text corpus.
At test time, we exploit the same beam search as the transducer, with the modification that the output label probabilities P(token|audio, prev_tokens) do not depend on the prev_tokens.
Two regularisers were used in this paper: early stopping and weight noise (the addition of Gaussian noise to the network weights during training).


# The RATS Collection: Supporting HLT Research with Degraded Audio Data
We introduce the RATS data collection that was designed to cover a diverse range of radio conditions. In the conditions of interest, the signal-to-noise ratio (SNR) often falls below 10dB.
The dataset includes four corpora, one for each of the RATS research tasks (Speech Activity Detection, Language Identification, Speaker Identification and Key Word Spotting), comprising clean source audio, the corresponding sets of 8 transceiver channels, and all channel-aligned annotations.


# Towards End-To-End Speech Recognition with Recurrent Neural Networks
In ASR, modelling language separately from sound is perhaps the most justifiable departure from end-to-end learning, since it is easier to learn linguistic dependencies from text than speech. Nonetheless, with the advent of speech corpora containing tens of thousands of hours of labelled data, it may be possible to learn the LM directly from the transcripts.
We present and end-to-end ASR model that consists of the deep bidirectional LSTM and the CTC output layer. Such a model has been applied to character-level speech recognition before, however, the relatively shallow architecture used in that work did not deliver compelling results.
We propose the Expected Transcription Loss to train the network to directly optimise the WER, and, more general, to optimise the expected value of an arbitrary loss function L defined over output transcriptions. To do this, we use Monte-Carlo sampling to approximate both L and its gradient, and prove that the gradient is unbiased. In our sampling, we use the same (sampled) CTC alignment, so that the loss variance largely cancels out (which is crucial when optimising with stochastic gradient estimates).
For example, if the sampled alignment yields the character transcription "WTRD ERROR RATE", the gradient would encourage outputs changing the second output label to "O", discourage outputs making changes to the other two words and be close to zero everywhere else.
The vast majority of alignments drawn from a randomly initialised network will give completely wrong transcriptions, and there will therefore be little chance of altering the loss by modifying a single output. We therefore recommend that expected loss minimisation is used to retrain a network already trained with CTC, rather than applied from the start.


# Listen, Attend and Spell
We present Listen, Attend and Spell (LAS): a seq-to-seq model for ASR (fig. 1) that consists of an encoder RNN, which is named the listener, and a decoder RNN, which is named the speller.
Key to our approach is the fact that we use a pyramidal RNN model for the listener, which reduces the number of time steps that the attention model has to extract relevant information from.
Rare and OOV words are handled automatically, since the model outputs characters (instead of words).
The speller produces character sequences without making any independence assumptions between the characters, that is the key improvement of LAS over previous end-to-end CTC models. For example, for the phrase “triple a” the model produces both “triple a” and “aaa” in the top beams. A model like CTC may have trouble producing such diverse transcripts for the same utterance because of conditional independence assumptions between frames.
Without the attention mechanism, the model overfits the training data significantly - it memorizes the training transcripts without paying attention to the acoustics.
During infarence, the ground truth is missing and the predictions can suffer because the model was not trained to be resilient to feeding in bad predictions at some time steps. To ameliorate this effect, we use a trick that was proposed in "Scheduled sampling for sequence prediction with recurrent neural networks": during training, instead of always feeding in the ground truth transcript for next step prediction, we sometimes sample from our previous character distribution and use that as the inputs in the next step predictions. We do not use a schedule and simply use a constant sampling rate of 10% right from the start of training.
We attempted to use the phonemes as a joint objective target, but found no improvements. We also attempted to pretrain the Listen function with context independent or context dependent phonemes generated from a conventional GMM-HMM system, but found no improvements.
Decoding is performed with a simple left-to-right beam search. A dictionary can optionally be added to constrain the search space to valid words, however we found that this was not necessary since the model learns to spell real words almost all the time.
We have vast quantities of text data, compared to the amount of transcribed speech utterances. We can use language models trained on text corpora alone similar to conventional speech systems. To do so we can rescore our beams with the LM. We find that our model has a small bias for shorter utterances so we normalize our probabilities by the number of characters in the hypothesis and combine it with a LM probability (eq. 16), where LM weight can be determined by a held-out validation set.


# Librispeech: An ASR corpus based on public domain audio books
We introduce the LibriSpeech corpus for ASR that is derived from audiobooks and is a part of the LibriVox project
It contains 1000 hours of speech sampled at 16 kHz
We automatically aligned the audio recordings with the corresponding texts, and split them into short segments
We tried to exclude segments of audio that might not correspond exactly with the aligned text
Models trained with our corpus do better on the standard Wall Street Journal (WSJ) test sets than models built on WSJ itself


# End-to-End Attention-based Large Vocabulary Speech Recognition
We propose a new approach to ASR, when alignment between the input features and the desired character sequence is learned automatically by an attention mechanism built into the RNN.
Training on long sequences can be made feasible by limiting the area explored by the attention to a range of most promising locations ("windowing"). This reduces the total training complexity from quadratic to linear.
We introduce a recurrent architecture that successively reduces source sequence length by pooling frames neighboring in time.
Integrating an n-gram language model into the decoding process yields recognition accuracies similar to other HMM-free RNN-based approaches.


# Wav2Letter: an End-to-End ConvNet-based Speech Recognition System
We present Wav2Letter: a model for end-to-end speech recognition that consists of CNN acoustic model
We train with an Auto Segmentation Criterion (ASG), an our alternative to the Connectionist Temporal Classification (CTC) (fig. 2, 3). In contrast to CTC, 1) there are no blank labels, and therefore t produces a much simpler graph, 2) we have un-normalized scores on the nodes, 3) we apply global normalization instead of per-frame normalization. We show that ASG can be faster than CTC, and as accurate.
We perform inference with a simple beam search decoder with beam threholding, histogram pruning and language model smearing
Our model shows competitive results in WER on the Librispeech corpus with MFCC features, and promising results from raw waveform.


# Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning
In ASR, the attention model has often been shown to improve the performance over CTC, mainly because it explicitly uses the history of the target character without any conditional independence assumptions. However, in realenvironment speech recognition tasks, the model shows poor results because the alignment estimated in the attention mechanism is easily corrupted due to the noise. Another issue is that the model is hard to learn from scratch due to the misalignment on longer input sequences, and therefore a windowing technique is commonly used to limit the area explored by the attention mechanism, but several parameters for windowing need to be determined manually depending on the training data.
We propose to use a shared-encoder representation trained by both CTC and attention model objectives simultaneously. Along with improving performance, our framework significantly speeds up learning with fast convergence.


# Far-Field ASR Without Parallel Data
When parallel audio recordings (close-talk microphone +  distant microphone) are available, the alignments (matchings between audio features and phonemes) used for training the acoustic models can be generated from close-talk microphone audio recordings to obtain WER improvements. However, far-field audio is usually not accompanied with close-talk microphone recordings.
We use the lattice-free maximum mutual information (MMI) objective (not proposed by us), which is tolerant to minor mis-alignment errors (such as shown in fig. 1), which is actual when alignments are generated from distant microphone recordings.
We also propose a method to select reliable utterances for training from distant microphone recordings.
These methods reduce the performance gap between the ASR systems that are trained using alignments generated from distant and close-talk microphone readings from 8% to 1.5%.
IMO, not clear why do we need alignment, while we can train with CTC loss.


# Joint CTC/attention decoding for end-to-end speech recognition
We propose a joint decoding algorithm for end-to-end ASR with a hybrid CTC/attention architecture.
Our joint CTC/attention approach combines the CTC and attention-based sequence probabilities in the inference step, as well as the training step.
The decoding objective is defined using multiplying text probabilities from CTC and attention (eq. 14). The CTC probability enforces a monotonic alignment that does not allow large jumps or looping of the same frames.
We perform one-pass/rescoring joint decoding, in which we compute the probability of each partial hypothesis using CTC and an attention model.
This greatly reduces irregular alignments without any heuristic search techniques.


# An analysis of incorporating an external language model into a sequence-to-sequence model
Recently the end-to-end LAS (Listen, Attend, and Spell) model was proposed for ASR. It  jointly learns an encoder, which serves as an acoustic model, a decoder, which serves as a language model, and an attention mechanism, which learns alignments.
Our goal is to explore why the performance of LAS still lags behind a SOTA ASR system with separate acoustic, pronunciation and language models. We propose that one reason for the performance degradation could be that the LAS decoder, that is trained only on transcribed audio-text pairs. In comparison, SOTA LMs are typically trained on a billion words or more.
We investigate the impact of training a separate LM on auxiliary text-only data, and incorporating this model as an additional cost term when decoding a LAS model (shallow fusion).
We find that RNN LMs are more effective at reducing error than n-gram LMs.
On Google Voice Search (which has much more training data than WSJ used in previous studies), we demonstrate that the use of shallow fusion with an neural LM with wordpieces yields a large WER reduction, obviating the need for second-pass rescoring, despite being 70 times smallerthan the second pass LM.


# Generation of Large-Scale Simulated Utterances in Virtual Rooms to Train Deep-Neural Networks for Far-Field Speech Recognition in Google Home
We develop an acoustic room simulator to generate large-scale simulated data for far-field speech recognition.
The system simulates millions of different room dimensions, a wide distribution of reverberation time and signal-to-noise ratios, and a range of microphone and sound source locations.
The simulator-driven approach is quite effective in obtaining large improvements in real / rerecorded conditions.


# Acoustic-To-Word Model Without OOV
Problem: the word-based CTC is a very good end-to-end ASR model, but it maps all the unknown words into OOV
We propose a hybrid CTC with both word-based CTC and character-based CTC heads that are synchronized
Whenever the ASR model emits an OOV token, we rely on character-based CTC


# Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models
We explore training attention-based seq2seq ASR models to directly minimize expected WER, instead of cross-entropy loss.
Our loss function is the expected number of word errors over the training set. We can approximate the expectation using an empirical average over samples drawn from the model. Its gradient can be itself be expressed as an expectation, which allows it to be approximated using samples. So, we approximate the WER expectation using N-best hypotheses decoded from the model using beam-search.


# Optimizing expected word error rate via sampling for speech recognition
IN ASR task Minimum Bayes risk (MBR) training have been shown effective in terms of WER. MBR minimizes an expected distance between a reference and a hypothesis. In word-level edit-based MBR, the distance between a reference and a hypothesis is measured as WER (given the prevalence of WER as an evaluation metric). However, this is hard to compute.
We show that the gradient of the expected loss optimized by word-level edit-based MBR training may itself be written as an expectation, allowing the gradient to be approximated by sampling.
The loss computation is shown on fig. 1. Sample_path is a sample from the model, collapse_path is a function that translates sample to output sequence (for CTC head, it removes all duplicates and blank tokens), get_loss computes edit distance (TODO what is gamma)?
IMO, looks similar to "Minimum word error training of long short-term memory recurrent neural network language models for speech recognition" and "Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models" and.


# Hybrid CTC/Attention Architecture for End-to-End Speech Recognition
We propose MOL: a hybrid CTC/attention end-to-end ASR. During training, we propose a multi-objective learning method by attaching a CTC objective to an attention-based encoder network as a regularization. This greatly reduces the number of irregularly aligned utterances. During decoding, we propose a joint decoding approach, which combines both attention-based and CTC scores in a rescoring/one-pass beam search algorithm to eliminate the irregular alignments.
Comparing with attention-only ASR, our model learned the desired alignment in an early training stage. This result indicates that the CTC loss guided the alignment to be monotonic.
This paper is a combination of two previous papers from the same authors "Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning" and "Joint CTC/attention decoding for end-to-end speech recognition" and extends them by providing more details and experimental discussions.


# Data augmentation improves recognition of foreign accented speech
We reproduce two accents, Latin American and Asian accented English speech with voice transformation (modifying glottal source and vocal tract parameters), noise addition, and speed modification.
We find that all augmentations provide improvements in accented ASR, with the largest gains coming from speed modification, then voice transformation and noise addition providing the least improvement.


# Toward domain-invariant speech recognition via large scale training
A problem: when ASR is used in conditions that do not match the training domain, performance significantly drops.
We combine large scale training data from multiple application domains, obtaining 162K hours of speech, and simulate conditions like background noise, codecs and sample rates.
We train a model that is robust to multiple application domains, and variations like codecs and noise, and allows for rapid adaptation for unseen conditions (using as little as 10 hours of data from a new domain - and performing on par with domain specific model trained from scratch using 70 times as much data).


# Representation Learning with Contrastive Predictive Coding
We propose a Contrastive Predictive Coding for unsupervised learning and demonstrate its effectiveness on four distinct domains: speech, images, text and reinforcement learning in 3D environments.
We stand for unsupervised pre-training, since it learns more general features than supervised pre-training. For example, in ASR pre-training features that are useful to transcribe human speech may be less suited for speaker identification, or music genre prediction. So, ASR pre-trained features will lack certain information. Same for image pre-training.
One way for unsupervised learning is to predict future observations (predictive coding).
We propose to compress raw data into a latent embedding space and train to predict future in this space (fig. 1) with autoregressive models. We rely on Noise-Contrastive Estimation for the loss function.


# Common Voice: A Massively-Multilingual Speech Corpus
We present the Common Voice: a massively-multilingual collection of transcribed speech
Over 50,000 individuals have participated so far, resulting in 2,500 hours of collected audio
Using either the Common Voice website or iPhone app, contributors record their voice by reading sentences displayed on the screen. The recordings are later verified by other contributors.
For languages with more than 500,000 Wikipedia articles, text sentences are extracted from Wikipedia using community provided rule-sets per language.


# Effectiveness of self-supervised pre-training for speech recognition
We compare different approaches of self-supervised pre-training for speech data
As one alternative (fig. 1a), we take "Discrete BERT" ASR model, that consists of vq-wav2vec, which quantizes the Librispeech dataset into 13.5k unique codes, follwed by pre-trained BERT model. Instead of vq-wav2vec we also tried k-means clustering MFCC and FBANK features with 13.5k centroids (to match the vq-wav2vec setup). We directly fine-tune the pre-trained BERT model on transcribed speech data using a CTC loss.
As another alternative (fig. 1b), we try "Continuous BERT". MLM cannot be performed with continuous inputs and outputs, as there are no targets to predict in place of the masked tokens. Instead, we classify the masked positive example among a set of negatives with InfoNCE loss. In this case, the inputs to BERT are dense wav2vec features, MFCC or FBANK features.
We show that the most effective method is to first learn a discrete vocabulary of the data with vq-wav2vec followed by standard BERT training over these discrete units. This performs much better than directly learning from the continuous audio data. Thus, disentangling acoustic unit discovery from learning the sequential relationship between them enables better representations.


# vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations
We propose vq-wav2vec, that learns discrete representations of fixed length segments of audio signal by utilizing the wav2vec loss and architecture.
To choose the discrete variables, we consider a Gumbel-Softmax quantization or K-means vector quantization. These methods perform relatively comparably.
Discretization enables the direct application of algorithms from the NLP community which require discrete inputs. We apply BERT to the quantized representations, and then pass BERT's outputs to any acoustic model such as wav2letter. We improve SOTA on the WSJ and TIMIT benchmarks by leveraging BERT pre-training.
We plan to explore self-supervised pre-training algorithms which mask part of the continuous audio input, or fine-tune BERT to output ranscriptions instead of feeding the pre-trained features to an acoustic model.


# The Spoken Wikipedia Corpus collection: Harvesting, alignment and an application to hyperlistening
The Spoken Wikipedia project unites volunteer readers of Wikipedia articles (for for persons who are unable or unwilling to read out of alexia, visual impairment, or because their sight is currently occupied, e.g. while driving). We present our open-source software pipeline that downloads, extracts, normalizes and text–speech aligns the Spoken Wikipedia.
We present and analyse, for three languages (de, en, nl), the resulting corpora of read encyclopedic content read by a large variety of speakers.


# An Unsupervised Autoregressive Model for Speech Representation Learning
We propose Autoregressive Predictive Coding (APC) for unsupervised speech representation learning.
We introduce a time shifting factor that asks the model to predict further steps. Our results show that the number of steps to the target frame controls what is learned in the representation. How this hyperparameter is set depends on how the representation is going to be used.
APC and CPC (Contrastive Predictive Coding) differ significantly in the type of information the model learns. (the difference is not clearly described in the paper; concurrent works?)


# A spelling correction model for end-to-end speech recognition
Usually acoustic models are accompanied with language models, but they are only trained on transcribed audio-text pairs, which leads to performance degradation especially on rare words.
When, on the contrary, we try to incorporate an external LM, we still observe that numerous rare word and proper noun errors. We hypothesize that this is because the LM was not trained with objective of correcting errors.
We propose the following method: given texts, we synthetically generate audio, run the baseline LAS speech recognizer on it, thus creating a set of textto-text pairs representing an error hypothesis and its corresponding ground truth. Then we train a spelling corrector (SC) model on these text-to-text pairs.
Our SC model is based on encoder-decoder bi-directional LSTM with attention.
During inference, acoustic model (LAS) with beam search produces N hypotheses with corrsponding log probability scores, and for every hypothesis our SC model can similarly be used to generate M new hypotheses with corresponding log probability scores. Rescoring all N×M candidates with an LM gives a set of LM scores. Finally, we can find the most likely hypothesis.


# Jasper: An End-to-End Convolutional Neural Acoustic Model
We propose Jasper: an end-to-end CNN that achieves SOTA on LibriSpeech among models without any external training data.
We propose a new residual connection topology we call Dense Residual (DR).
We use Connectionist Temporal Classification (CTC) loss.
To improve training, we further introduce a new layer-wise optimizer called NovoGrad, a variant of the Adam with a smaller memory footprint.


# Universal Adversarial Perturbations for Speech Recognition Systems
We propose an algorithm to find a single quasi-imperceptible perturbation, which when added to any arbitrary speech signal, will most likely fool  a victim ASR model.
The algorithm requires access to the victim’s model architecture and parameters.
Attack Success Rate depends on the magnitude of the perturbation (fig. 3).
Perturbations generalize to a significant extent across models that are not available during training.


# SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
We propose SpecAugment: an augmentation that is applied to the feature inputs of a NN (log mel spectrogram).
SpecAugment consists of warping the features, masking blocks of frequency channels, and masking blocks of time steps.
SpecAugment greatly improves the performance of ASR networks.


# CSS10: A Collection of Single Speaker Speech Datasets for 10 Languages
The existing multi-lingual speech-text datasets have several problems: 1) the Tundra dataset uses only one audiobook per language, 2) The M-AILABS uses multiple speakers in each language, which make it unideal for the single speaker TTS task (namely, generating speech from text in the voice of a single speaker) where more data from a single speaker tends to help model performance.
We propose CSS10, a collection of single speaker (one speaker per language) speech datasets for ten languages, composed of short audio clips from LibriVox audiobooks.


# Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks
We propose problem-agnostic speech encoder (PASE) for self-supervised speech models, where a single feature encoder is followed by multiple workers that solve different self-supervised tasks, defined as regression or binary classification (fig. 1). The intuition is that each self-supervised task may bring a different view or soft constraint on the learned representation.
We employ simple worker structure to encourage the encoder, and not the workers, to discover high-level features.


# wav2vec: Unsupervised Pre-training for Speech Recognition
We propose wav2vec model that was trained on large amounts of unlabeled audio
Wav2vec takes raw audio as input and computes a general representation at a lower temporal frequency (fig. 1). The encoder is a 5-layer CNN, and the output stride is 10 ms, and the receptive field is about 30 ms of 16 kHz. Then, the 9-later CNN "context network" mixes multiple representations with a total receptive field about 210 ms (810 ms for large model).
The objective is to predict future samples from a given signal context, with contrastive loss that requires distinguishing a true future audio sample from negatives (as in "Representation Learning with Contrastive Predictive Coding").
To fine-tune wav2vec for the TIMIT task (predicting phonemes), we pass output representations (instead of MFCC features as a baseline) into acoustic CNN model, which outputs phoneme probabilities. The model is trained using the Auto Segmentation Criterion.
To fine-tune wav2vec on WSJ benchmark (transribing text), acoustic CNN model predicts probabilities for 31 graphemes, including the standard English alphabet, the apostrophe and period, two repetition characters (e.g. the word ann is transcribed as an1), and a silence token used as word boundary. For decoding the emissions from the acoustic model we use a lexicon as well as a separate language model trained on the WSJ language modeling data only.


# Towards a Competitive End-to-End Speech Recognition for CHiME-6 Dinner Party Transcription
End-to-end ASR models are prone to accuracy degradation in noisy and low-resource conditions.
We show that on the CHiME-6 Challenge data (real dinner parties recorded in reverberant and noisy conditions), our best end-to-end model (RNN-Transducer with improved beam search and the Guided Source Separation augmentation) outperforms the hybrid baseline (TDNN-F, factorized time-delayed neural network) system only by 2.7% WER.


# Conformer: Convolution-augmented Transformer for Speech Recognition
We propose Conformer, convolution-augmented transformer for speech recognition
Each conformer block contains convolution and MHSA
We use a single-LSTM-layer decoder in all our models
We perform various ablation studies
A concurrent work with wav2vec 2.0


# Learning Robust and Multilingual Speech Representations
We learn audio representations using contrastive predictive coding, then we train an ASR model using our representations, and evaluate it under domain and language shifts.
We find that large pretraining dataset lead to more robustness to domain shifts, compared to both log filterbank features as well as to pretraining just on LibriSpeech.
We also train ASR models on 25 languages, and show that our representations outperform those pretrained only on clean English data in the language transfer setup.
We confirm that usupervised representations consistently improve robustness on downstream tasks, and representations learned from multilingual data can transfer across many languages.


# Attentional Speech Recognition Models Misbehave on Out-of-domain Utterances
We discovere the problem of echographic transcription: when autoregressive seq2seq models are used to decode out-of-domain audio, the output transcript contains the same words or phrases repeated over and over again. There are many 5-second recordings that produce more than 500 characters of decoding output.
When decoding audio from the British National Corpus with an attentional encoder-decoder model trained solely on the LibriSpeech corpus.
This behavior occurs even when the model performs well on the in-domain task.
A frame-synchronous hybrid (DNN-HMM) model trained on the same data does not produce these unusually long transcripts (fig. 1), but these decoding issues are reproducible in a speech transformer model from ESPnet, and to a lesser extent in a self-attention CTC model. This suggests that these issues are intrinsic to the use of the attention mechanism, when the decoder can attend over the entire length of the encoded input to generate each output token.
When the outputs are very repetitive, the attention mechanism of the AR-S2S model attends to the same section of audio without proceeding forwards in time (fig. 2).
We create a separate length prediction model to predict the correct number of wordpieces in the output, which allows us to identify and truncate problematic decoding results.


# Racial disparities in automated speech recognition
ASR systems exhibit WER of 0.35 for black speakers compared with 0.19 for white speakers.
We suggest that ASR systems are confused by the phonological, phonetic, or prosodic characteristics of African American Vernacular English.
The likely cause of this shortcoming is insufficient audio data from black speakers when training the models.


# Rethinking Evaluation in ASR: Are Our Models Robust Enough?
We study acoustic model transfer across five public datasets, as well as transfer to out-of-domain, real-world audio data.
No single validation or test set from public datasets is sufficient to measure transfer to other public datasets or to real-world audio data. This s uggests that ASR researchers interested in producing transferable acoustic models should report results on several public datasets, at very least including TED-LIUM (v3).
Reverberative and additive noise augmentation improves generalization performance across domains.
We provided a recipe for a community-reproducible robust ASR model, which can be trained with a couple of public audio datasets, and language models trained on the Common Crawl dataset


# ASR Error Correction and Domain Adaptation Using Machine Translation
We propose to carry out ASR error correction via domain adaptation. We learn an adaptation module that goes from hypothesis of pre-trained ASR towards reference text (as in automatic machine translation), thus learning to fix systematic errors the pre-trained ASR makes due to domain mismatch.


# The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling
We introduce a new unsupervised task, spoken language modeling: the learning of linguistic representations from raw audio signals without any labels. We suggest that self-supervised acoustic models may actually go beyond acoustic modeling, learning their own LM from raw audio.
We introduced the new Zero Resource Speech Benchmark 2021 for spoken language models. It is composed of 4 zero-shot tests probing 4 linguistic levels: acoustic, lexical, syntactic and semantic. They are zero-shot in that they do not require training a classifier.
A self-supervised pipeline of Contrastive Predictive Coding + k-means clustering + LM (LSTM or BERT), trained on LibriSpeech, can perform above chance on all of these tests, while being worse than text-based models trained on the same data.
Seems like there is another close paper (another version?): https://arxiv.org/abs/2102.01192v2


# MLS: A Large-Scale Multilingual Dataset for Speech Research
We intoduce the Multilingual LibriSpeech (MLS) dataset derived from read audiobooks from LibriVox and consists of 8 languages, including about 44.5K hours of English and a total of about 6K hours for other languages


# Multi-task self-supervised learning for Robust Speech Recognition
We propose PASE+, an improved version of PASE for self-supervised speech models, to perform better in noisy and reverberant environments. We employ an online speech distortion module, that contaminates the input signals with various disturbances (fig. 1). We also combine our CNN encoder with a quasi-recurrent neural network (QRNN). Finally, we refine the set of self-supervised workers.


# Unsupervised pretraining transfers well across languages
We show that a slight modification of the CPC (contrastive predictive coding) pretraining extracts features that transfer well from  English (Librispeech) to several low-resource languages from the Common Voice database.
Out modifications: 1) we replace batch normalization with a channel-wise normalization, 2) we replace each linear classifier with a 1-layer Transformer network, 3) we use an LSTM instead of a GRU
To evaluate on a target language, we freeze the model after the pre-training and simply learn a linear classifier for the targeted language, using CTC loss. This procedure explicitly measures the linear separability of the phoneme representation, once transferred to a target language.


# A Streaming On-Device End-to-End Model Surpassing Server-Side Conventional Model Quality and Latency
For ASR on-device streaming models, we develop a first-pass RNN-T model and a second-pass LAS rescorer fo achieve both high quality and low latency (the delay between when a user stops speaking and the hypothesis is finalized).
We combine these models because RNN-T models have been shown to be competitive in quality, and non-streaming models, such as LAS, have been shown to perform well nder low-latency constrains.
We train our model on multi-domain audio-text utterance pairs, including search traffic, telephony data and YouTube data to increase acoustic diversity and the vocabulary seen by the model. We also train with accented English speech to make the model more robust to different pronunciations.
One of the issues with using multi-domain data is that each domain has different transcription conventions ("$100" versus "one hundred dollars"). We explore feeding a domain ID to the RNN-T encoder as a one-hot vector, with the UD being one of the 4 domains. Thus we are able to improve upon a model trained on voice search data only (single domain).
We also explore various ideas to improve latency of our model.


# Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss
We propose Transformer Transducer by replacing ENN-based audio and label encoders in the RNN-T architecture with Transformer encoders. As in the original RNN-T model, the joint network (fig. 1) at each step combines the audio encoder output and the label encoder output given the previous non-blank output token sequence. The joint network returns the distribution over the next token. Our model has 18 audio and 2 label encoder layers.
The model uses the RNN-T loss (see the original paper "Sequence Transduction with Recurrent Neural Networks")
Our model achieves a new SOTA on the LibriSpeech benchmark


# Using Synthetic Audio to Improve The Recognition of Out-Of-Vocabulary Words in End-To-End ASR Systems
We aim to boost the recognition accuracy of RNN-T model on out-of-vocabulary (OOV) words.
We use a text-to-speech (TTS) engine to provide synthetic audio. We use 2.3K hours of anonymised far-field in-house data. OOV words are all the words that have not appeared in train set but have appeared at least three times in dev set. The utterances in Dev containing any OOV words are extracted as a subset DevOOV (this only accounts for 0.7% of the Dev set, or 6.5K utterances). Similarly, the utterances in Eval containing any (the same) OOV words are extracted as a subset EvalOOV, containing 4.3K utterances.
The best performance is achieved by fine-tuning the RNN-T on both original training data and extra synthetic data with elastic weight consolidation (EWC) applied on the encoder. This yields a 57% relative word error rate (WER) reduction on utterances containing OOV words without any degradation on the whole test set.


# XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale
-


# A Toolbox for Construction and Analysis of Speech Datasets
The creation of new speech datasets remains an ongoing problem, because 1) Domain lexicon and language changes over time, 2) datasets need to contain a variety of samples with noisy room conditions, multi/single speakers recordings, speakers with different geographic origin, accent, gender, age, etc.
We introduce an open-sourced NeMo toolbox for analysis of existing speech datasets and construction of new speech corpora (fig. 2). The CTC-Segmentation tool can be used to splitting long audio files, and Speech Data Explorer (SDE) tool is for interactive audio data analysis.
We use our toolbox explaining how to construct the Russian LibriSpeech corpus, which improved WER on the MCV Russian dev subset.


# English Accent Accuracy Analysis in a State-of-the-Art Automatic Speech Recognition System
We train an ASR model on the MLS dataset, and test the model on the CommonVoice dataset, which has labels indicating accent.
We see that WER can degrade up to an absolute 10% for accents with phonetic and prosodic characteristics further from American and UK English, like Asian accents.


# SpeechStew: Simply Mix All Available Speech Recognition Data to Train One Large Neural Network
We present SpeechStew, an ASR model that is trained on a combination of various publicly available ASR datasets.
SpeechStew uses the Conformer RNN-T architecture.
SpeechStew simply mixes all the datasets without any balancing.
SpeechStew achieves SoTA or near SoTA results across a variety of tasks, without the use of an external LM, and learns powerful transfer learning representations.
On noisy, low resource ASR datasets, such as CHiME-6, end-to-end methods struggle relative to HMM-based baselines. We show that simple fine-tuning SpeechStew on CHiME-6 without a LM give WER compareble to a strong HMM baseline with LM. So, one can simply finetune a pre-trained model for only a few thousand gradient steps and achieve strong results.


# WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
We propose WavLM, a self-supervised model that jointly learns masked speech prediction and denoising.
Speech denoising allows to apply the model to non-ASR tasks, such as diariazation, separation, speech enhancement.
WavLM employs gated relative position bias to better capture the sequence ordering of input speech.
We use more training data to eliminate the audiobook data bias (60k hours of Libri-Light, 10k hours of GigaSpeech, and 24k hours of VoxPopuli)


# W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training
We propose w2v-BERT that explores MLM for self-supervised speech representation learning.
w2v-BERT combines the core methodologies from wav2vec 2.0 and BERT (fig. 1). It discretizes input continuous speech signals into a finite set of discriminative speech tokens with contrastive learning, and solves a masked prediction task on the discretized tokens. In contrast to HuBERT which relies on an iterative re-clustering and re-training process, or vq-wav2vec, which concatenates two separately trained modules, w2v-BERT directly optimizes a contrastive loss and a masked prediction loss simultaneously. In turn, wav2vec 2.0 only employs contrastive learning, whose resulting ASR performance lags behind that of combining contrastive learning and masked prediction.
We pre-train w2v-BERT on 60k hours of unlabeled speech data from the Libri-Light corpus
w2v-BERT yields SOTA performance on the well-benchmarked LibriSpeech task, after fine-tuning on LibriSpeech by adding an LSTM decoder, so that the ASR network is a sequence transducer.


# Quantifying Bias in Automatic Speech Recognition
We systematically quantify the bias of a Dutch SotA ASR system against gender, age, regional accents and non-native accents.
Female speech is better recognised than male speech, for all native and non-native groups and for both speech styles.
Among the native speakers, teenagers achieve the best WER performances in read and HMI speech, followed by the older adults;(age 65+) (their speech was not always well articulated) while children were the worst recognised. Speech of native speakers is recognised much better than that of non-native speakers of Dutch. Among the non-native speakers, the performance differences between children and adults do not differ much.
For each group, the WER performance of human-machine interaction speech is consistently worse than that of read speech, probably because the former is less well prepared than the latter. This confirms that the size of the bias is influenced by the speaking style of the person.
As for accents, speech spoken by people from Flanders achieved the worst WER performance in all age groups except for the older adults. Also, the results suggest that older speakers in the Netherlands typically have stronger regional accents than children and teenagers.


# The People's Speech: A Large-Scale Diverse English Speech Recognition Dataset for Commercial Usage
We introduce the People’s Speech, a supervised ASR dataset. In total, there are 52,500 hours of audio in our dataset, of which 51,890 hours are English. The dataset also includes 23 other languages (fig. 2). The dataset contains different background noise (fig. 6)
The data is collected via searching vimeo.com and archive.org for appropriately licensed audio data with existing transcriptions and aligning the audio to the transcripts.
Audio content with transcripts on the Internet Archive is 1) abundant, 2) searchable by license type, and 3) diverse, facilitating the creation of a our large-scale open speech dataset, so large-scale datasets can be created more efficiently than previously thought.
We do not create train, test, and dev splits for our dataset for two reasons. One, we don’t have a way to ensure there is no speaker overlap between the splits. Two, we don’t have a way to ensure duplicated audio data.


# Multi-Variant Consistency based Self-supervised Learning for Robust Automatic Speech Recognition
A problem in self-supervised audio models is that the clean signal and the background noise play the same role in the self-supervised objective function. If the signal-to-noise ratio (SNR) is low, the SSL model may pay more attention to the inessential background noise than the text-related speech signal.
We propose a multivariant consistency based self-supervised learning (MVC-SSL) as a robust pre-training method that adapts to different environments. This method is designed for noisy and distant-talking speech in real-world applications.
MVC-SSL can calculate the contrastive loss among audios from different channels or acoustic conditions. As a result, the MVC-SSL model can be more robust with the background noise and reverberation.
We evaluate on the CHiME-4 and AMI. The proposed method can reduce 30% relative WER over the baseline wav2vec2.0.


# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units
Self-supervised pre-training for speech is complex, because the boundaries between sound units are not known, there is no prior lexicon of discrete sound units available, and multiple sounds may be present in each input utterance.
We propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning.
HuBERT relies on predicting K-means cluster assignments of masked segments of continuous input. HuBERT utilizes an offline clustering step to provide aligned target labels for a BERT-like prediction loss (fig. 1).
The learned representation quality improves dramatically with iteratively refining K-means cluster assignments using learned latent representations for a previous iteration. Since we expect a pre-trained model to provide better representations than the raw acoustic feature such as MFCCs, we can create a new generation of clusters by training a discrete latent model (such as K-Means or GMM) over the learned latent representations. This idea is similar to iterative pseudo labeling for semi-supervised ASR.
HuBERT either matches or improves upon the state-of-theart wav2vec 2.0 performance on all fine-tuning subsets of 10mins, 1h, 10h, 100h, and 960h.
IMO, the refinment stage is not explained clearly. Probably, they run K-Means over the feature vectors obtained on the last layer.
IMO this looks like a denoising and discretizing autoencoder, that will capture not only phonemes, but many irrelevant (for many tasks) details, such as voice identity, pace, white and structured noise and background speech.


# DF-Conformer: Integrated architecture of Conv-TasNet and Conformer using linear complexity self-attention for speech enhancement
We focus on Single-channel speech enhancement (SE), the task of recovering target speech from a noisy signal (which does not include interference speech signals). Conv-TasNet is a powerful model for SE. One of the main research topics in SE is improving the mask prediction architecture ("M" in eq. 1). For example, the improved time-dilated convolution network (TDCN++) extended Conv-TasNet to improve SE performance. A promising candidate for improving mask prediction networks is the Conformer architecture.
We propose DF-Conformer (dilated FAVOR Conformer). We combine Conformer layers with the dilated convolution layers of the TDCN++ architecture. To make the model computationally feasible, we use a linear-complexity variant of self-attention in the Conformer, known as fast attention via positive orthogonal random features (FAVOR+), as used in Performer.
To train the model, we use speech from LibriVox and non-speech sounds from freesound.org.
Examples of spectrograms and attention matrices in DF-Conformer-8 are shown in fig. 2.


# PARP: Prune, Adjust and Re-Prune for Self-Supervised Speech Recognition
We first show that applying widely-adopted pruning methods, such as One-Shot Magnitude Pruning (OMP) and Iterative Magnitude Pruning (IMP) for SOTA speech SSL models is extremely time-consuming, and gives no performance boost.
We present Prune-Adjust-Re-Prune (PARP), a method for discovering sparse subnetworks within pre-trained speech SSL. It consists of the following two cyclic steps: 1) Prune the model, 2) Fine-tune the network on target downstream task/language while zeroing out pruned weights, but allowing them to be be updated by gradient descent. After a few number of model updates, re-prune again and so on. So, different from other pruning methods, PARP allows pruned-out weights to be revived during finetuning.
We fine-tune we pre-trained wav2vec2-base and wav2vec2-large on the 10h/1h/10min splits from Librispeech and Libri-light and show that PARP yields over 10% WER reduction over the full model under ultra-low data regime (fig. 3), and the method adds minimal computation overhead to existing SSL downstream fine-tuning.
We also show the transferability of pruning mask discovered from a source language by finetuning its subnetwork on another target language, and the possibility of obtaining a single shared subnetwork for all downstream languages.
We also demonstrate PARP’s effectiveness on pre-trained BERT/XLNet, mitigating the cross-task performance degradation reported in BERT-Ticket.
IMO, fine-tuning SSL wav2vec2 model on LibreSpeech is very different from fine-tuning a pre-trained ASR model on some specific dataset. In the latter case we also need to compare the fine-tuned versions against the original model.


# Citrinet: Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition
End-to-end neural ASR models can be roughly classified into 1) non-autoregressive CTC models, 2) autoregressive Seq2Seq models with attention, 3) autoregressive RNN-Transducers. CTC models are more stable to train than autoregressive models, but the latest Seq2Seq models and RNN-Transducers significantly outperform them.
It is believed that "Because of the strong conditional independence assumption, CTC does not implicitly learn a LM over the data (unlike the attention-based encoder-decoder architectures). It is therefore essential when using CTC to interpolate a language model".
We propose Citrinet: a deep convolutional CTC model for ASR. It significantly closes the gap between CTC and the best Seq2Seq and Transducers models without any external LM, contrary to the common belief that "CTC requires an external LM to output meaningful results".


# Don't speak too fast: The impact of data bias on self-supervised speech models
We study how pre-training data affects self-supervised speech models by pre-training models on biased datasets targeting different factors of speech, including gender, content, and prosody.
Our experiments show that self-supervised speech models (TERA and APC) have tolerance toward gender bias.
We find that the content of speech has little impact on the performance of S3Ms across downstream tasks
self-supervised speech models do show a preference toward a slower speech rate.


# Layer-wise Analysis of a Self-supervised Speech Representation Model
We analyze wav2vec 2.0 representations.
We use projection-weighted Canonical Correlation Analysis to measure similarity between the wav2vec 2.0 layer representations and quantities of interest such as 1) representations from from adifferent layer of the same model, 2) representations from a fine-tuned version of the model, 3) mel filter bank features (acoustic information), 4) acoustically grounded word embeddings (word-level acoustic-phonetic information), 5) GloVe word embeddings (word meaning information).
We use mutual information to measure dependence between the the wav2vec 2.0 layer representations and the corresponding (discrete) phone or word labels.
As we go deeper into the model, the representation starts deviating from the input speech features followed by a reverse trend where even deeper layers become more similar to the input, as if reconstructing the input (fig. 1, black line).
The shallowest layers encode acoustic features, followed by phonetic, word identity, and word meaning information, and then followed by a reverse (fig. 1, other lines).
The final convolutional layers and initial transformer layers are highly correlated with mel spectrogram features, suggesting that the model learns to extract features similar to human-engineered ones.
Fine-tuning the model for ASR breaks the autoencoder-style behavior in the final few layers. Higher layers change the most in fine-tuning, suggesting that the pre-trained model may not serve as a good initialization of these top layers for ASR. This finding suggests re-initializing the final 1-3 layers before fine-tuning, as has been recently discovered for BERT. This outperforms the standard approach of initializing all layers from the pre-trained model, with large improvements when fine-tuning on the 10-minute training set and minor improvements for larger training sets.


# ASR4REAL: An extended benchmark for speech models
We introduce a set of ASR benchmarks matching real-life conditions.
We have noticed a small accuracy gap based on gender.
ASR models show big accuracy variations by accents.
ASR models show very strong performance gap based on the socio-economic status of the speaker.
ASR models show important performance drop onspontaneous speech.
LMs in their current form are not be adapted for spontaneous speech: even a LM trained on a dataset as big as Common Crawl does not seem to have significant positive effect.
This reiterates the importance of developing conversational LMs.


# Wav2vec-C: A Self-supervised Model for Speech Representation Learning
The wav2vec 2.0 problem formulation can result in several locally optimal codebooks. Some examples from our experiments: 1) only two codes are assigned: one for speech and the other for nonspeech; 2) the model assigns specific codes to fixed temporal locations, irrespective of the speech, to enable a good contrastive loss. The latter is especially probable if voice usually occurs at fixed temporal locations, as in our dataset. Hence, the wav2vec 2.0 codebook learning methodology might not generalize well to our pre-training dataset.
We propose Wav2vec-C - a modification of wav2vec 2.0, regularized by an additional consistency network that learns to reconstruct the input features from the quantized representations in a way similar to a VQ-VAE model. This enforces the latent space to preserve meaningful information that enable a low reconstruction error.
Our encoder network f (fig. 1) is a 3-layer LSTM as encoder with log-STFT input features. We quantize LSTM outputs with a product quantization module ("q" on fig. 1): we split each 768-demensional output vector into a pair of 384-dimensional sub-vectors, and store two codebooks (one for each sub-vector). Each codebook contains 320 384-dimensional code vectors, and quantization is performed by nearest neighbor search. Then the two code vectors are concatenated. To learn codebooks, we use either Gumbel-softmax, or K-means with a specific backprop method (sec 2.4.1, 2.4.2). We use a SpecAugment module to mask out portions of the continuous encodings, and feed them into a transformer context network g (fig. 1) with sinusoidal positional embedding, outputting the context representations. We apply a contrastive loss between the quantized encodings and the context representations, by selecting 50 negative samples. Our consistency network r (fig. 1) is a 3-layer LSTM that accepts the quantized embeddings and outputs "consistency vectors", denoted as S. We minimize the consistency loss: L2 distance between S and the log-STFT input features X. If we ignore the consistency loss, the rest of the model is similar to wav2vec 2.0 with one difference: wav2vec 2.0 uses CNN encoder that accepts raw audio. We also tune a magnitude of a diversity loss in Gumbel-softmax to avoid the codebook collapse that is commonly observed in VQ problems.
For SSL and fine-tuning, we use real far-field speech with varied degrees of SNR ranging between -40 to 50 dB. The proposed SSL model after RNN-T fine-tuning achieved a 1.4% relative WER reduction over baseline (without SSL) compared to a 0.7% reduction from wav2vec 2.0.
We also observed that ASR robustness is correlated with codebook diversity. Our model with Gumbel-softmax codebook fully utilizes codebook, while wav2vec 2.0 on our data under-utilizes codebook. The k-means codebook uses only a small fraction of the codes. The k-means codebook is is better than Gumbel-softmax for noisy test sets, and the Gumbel-softmax codebook is better on clean test sets. We hypothesize that a small codebook diversity may be good for noise robustness.


# The Multilingual TEDx Corpus for Speech Recognition and Translation
We present the Multilingual TEDx corpus, a collection of audio recordings from TEDx talks in 8 source languages (es, fr, pt, it, ru, el, ar, de).


# Hallucination of speech recognition errors with sequence to sequence learning
We present a dual encoder model for ASR error prediction that can look at both word and phoneme sequence representations of input text to further improve the fidelity of hallucinated errors. In our model, the word sequence decoder is conditioned on a word sequence encoder and a phoneme sequence encoder from the ground truth transcription (fig. 1, 2).
This may be helpful when cascading ASR systems with NLU systems trained on text data. We can augment input text using our model, so that the downstream NLU model observes during training an input that contains the kind of errors expected from ASR at test time, and thus can learn to be robust to them.


# The Accented English Speech Recognition Challenge 2020: Open Datasets, Tracks, Baselines, Results and Methods
Accent poses a great challenge to the robustness of ASR. The difference between accents is mainly reflected in three aspects of pronunciation: stress, tone and duration.
We release a set of 160 hours of accented English speech from 8 countries with labels as the training set, and another 20 hours of speech without labels as the test set, including two unseen accents from another two countries.
Track 1 aims to study the English accent recognition problem. The final ranking is based on the recognition accuracy on the whole test set. The winner used TDNN based classification network with phonetic posteriorgram (PPG) feature as input, and they use text-to-speech (TTS) to expand the training data. Most teams have done a lot of work in data augmentation.
Track 2 studies the robustness of ASR system on accented English speech where the word error rate on the whole test set is used as the evaluation metric. Test sets include accents beyond training data in order to evaluate the generalization performance of the model. The winner used a CTC model with LAS rescoring while the CTC model was initialized by a Wav2Vec encoder trained in an unsupervised manner. The superior performance indicates that unsupervised training is promising in improving performance when labeled data is limited. Similarly to track 1, various data augmentation tricks were widely adopted, including volume augmentation and speed perturbation. Noise and reverberation did not give an obvious gain, probably because the acoustic and channel conditions between the test set and the training set is similar.


# Residual Adapters for Parameter-Efficient ASR Adaptation to Atypical and Accented Speech
We show that residual adapters work extremely well for acoustic adaptation of different speech models.
We study personalized models for atypical speech, and group models for accented speech.
Adapter layers work well for two different SOTA models: RNN-T, and Transformer Transducers (T-T).
This provides an easy way to deploy and store adapted models: a (generic) base model is deployed to all clients, and each individual or group can receive a personalized set of trained adapter layers that is small in size.


# Improving Noise Robustness of Contrastive Speech Representation Learning with Speech Reconstruction
Usually noise robustness in ASR is achieved by speech enhancement modules. However, the ASR performance of the noise-suppressed speech may be degraded, caused by the nonlinear distortion brought by DNN. One solution is to train both networks jointly to allow train-time adaptation of the ASR network to such distortions. This complicates architecture and training.
We propose a novel self-supervised approach to address the robust ASR problem. We combine a reconstruction module with the contrastive learning framework of wav2vec 2.0.
Our network (fig. 1) takes in a noisy waveform as the input. We optimize the network 1) to reconstruct the clean speech, and 2) to solve the contrastive task by distinguishing the positive samples from the negative samples. Our reconstruction module uses two-layer bidirectional LSTM network with layer normalization, followed by a CNN decoder. Reconstruction is only performed in the pre-training stage, and it is not required in fine-tuning and inference.
In comparison with Wav2vec-C, we explicitly incorporate the denoising process, while they mainly focused on improving the codebook utilization ratio of wav2vec 2.0.
On Librispeech with synthetic noise, by adding the reconstruction module, our proposed model maintained the performance for clean speech, meanwhile, performed significantly better for noisy speech. For the real-world noisy speech from the CHiME-4 challenge, we have obtained the state of the art ASR performance without any denoising front-end, but the performance boost for the LibriSpeech based model was more obvious.
IMO, in CV there were shown that the robustness to synthetic distribution shifts, including noise does not make the model to be robust to the real distribution shifts. However, maybe this does not hold in ASR.


# VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation
We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages from 2009-2020 European Parliament (EP) event recordings.
VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours.
We examine the out-of-domain pre-training on VoxPopuli with further few-shot phoneme recognition on CommonVoice. The results suggest the high generality of the speech representations learned from VoxPopuli.
Pre-training with in-domain VoxPopuli  unlabeled data substantially improves performance on VoxPopuli ASR data, especially for low-resource languages.


# SUPERB: Speech processing Universal PERformance Benchmark
We introduce Speech processing Universal PERformance Benchmark (SUPERB) as a challenge with a leaderboard and a benchmark.
SUPERB targets at the direct usability of pretrained models on various popular tasks through any usage. We focus on investigating a simple framework solving all SUPERB tasks with a frozen, shared pretrained model, and lightweight prediction heads finetuned for each task. The framework puts an explicit constraint on downstream models to be as lightweight as possible for all tasks.
SSL models explored in this paper are summarized in Table 1. They include wav2vec 2.0, HuBERT and many other models.
Tasks from ASR: Phoneme Recognition, Automatic Speech Recognition, Keyword Spotting (detects preregistered keywords), Query by Example Spoken Term Detection (detects a spoken term (query) in an audio database (documents) by binary discriminating a given pair of query and document into a match or not).
Tasks to analyze speaker modeling: Speaker Identification (classifies each utterance for its speaker identity, where speakers are in the same predefined set for both training and testing), Automatic Speaker Verification (verifies whether the speakers of a pair of utterances match as a binary classification) and Speaker Diarization (predicts who is speaking when for each timestamp, and multiple speakers can speak simultaneously).
Tasks from Spoken Language Understanding: Intent Classification (classifies utterances into predefined classes to determine the intent of speakers) and Slot Filling (predicts a sequence of semantic slot-types from an utterance, like a slot-type FromLocation for a spoken word Taipei, which is known as a slot-value).
Task from paralinguistics: Emotion Recognition (predicts an emotion class for each utterance).
Since the last-layer representation is not always the best, the framework collects multiple hidden states from the pretrained model and weighted-sum them as the final representation. PR, KS, SID, IC, ER are simple tasks that are solvable with linear downstream models. For ASR, a vanilla 2-layer 1024-unit BLSTM is adopted and optimized by CTC loss on characters. The trained model is decoded with LibriSpeech official 4-gram LM powered by KenLM. As for ASV, we adopt the well-known x-vector as the downstream model.
wav2vec 2.0 and HuBERT outperforms others by a large margin on Phoneme Recognition and Intent Classification tasks. Their results on Speaker Identification and Emotion Recognition are also highly competitive.
FBANK cannot work on any task using linear models, but achieves competitive performance when allowing non-linear downstream models (in Automatic Speech Recognition, Slot Filling, Automatic Speaker Verification, and Speaker Diarization).
We observe that the ranking on Phoneme Recognition aligns with Automatic Speech Recognition weakly.
IMO, while Emotion Recognition refers to paralinguistics, spoken words may also highly correlate with emotions, so this task has two ways to solve with different transferability to another conditions.


# Combining Spectral and Self-Supervised Features for Low Resource Speech Recognition and Translation
Spectral features (SF) such as log Mel-filterbanks could be more robust to domain shifts than self-supervised (SSL) features, since the majority of SSL models are trained using English speech only.
We combine SF and SSL representations through learnable fusions (linear, convolutional and co-attention based combinations).
We obtained strong improvements over ASR and ST datasets compared with the SSL baseline. Our models performances are strong for both in-domain and out-of-domain scenarios.
We also propose a MoE-based technique that enables quantifying the domain shift between the SSL training data and the target language resources. For example, Arabic model uses HuBERT and FBANK with similar weights, but Totonac (a rare language) model seems to rely at more than 80% on FBANK features.


# A Brief Overview of Unsupervised Neural Speech Representation Learning
We review the development of unsupervised representation learning for speech over the last decade.
Speech data offers unique challenges for unsupervised representation learning. As a result, methods from other domains rarely translate directly.
We group previous work into self-supervised models and probabilistic latent variable models (fig. 1). These categories are neither exhaustive nor mutually exclusive.
In general, work on global representations within self-supervised learning precedes work on local representations.
We also briefly touch upon evaluation procedures. A common approach is to evaluate the representations in terms of their usefulness for downstream tasks.


# AudioLM: a Language Modeling Approach to Audio Generation
A current problem is that when not provided with strong conditioning (e.g., linguistic features), current audio synthesis models generate unstructured audio, such as babbling speech.
We introduce AudioLM (fig. 1, 2), a framework for audio generation that enables high-quality audio generation with long-term coherent structure. Starting from raw audio waveforms, we first construct coarse semantic tokens. Autoregressive modeling of these tokens captures both local dependencies (e.g., phonetics in speech, local melody in piano music) and global long-term structure  (e.g., language syntax and semantic content in speech; harmony and rhythm in piano music). However, these tokens lead to poor reconstruction. So, in addition to semantic tokens, we rely on fine-level acoustic tokens produced by a SoundStream neural codec, which capture the details of the audio waveform. We train a LM to generate both semantic and acoustic tokens.
When conditioned on a prefix (or prompt) of only 3 seconds of speech from a speaker not seen during training, AudioLM produces consistent continuations while maintaining the original speaker voice, prosody and recording conditions.
AudioLM is also suited for music generation.
We show that the semantic tokens and he acoustic tokens complement each other in terms of phonetic discriminability and reconstruction quality.


# ASR data augmentation in low-resource settings using cross-lingual multi-speaker TTS and cross-lingual voice conversion
We propose an ASR data augmentaiton method suitable using only a single real speaker in a target language.
We perform augmentations with cross-lingual multi-speaker speech synthesis and cross-lingual voice conversion (fig. 1).
We used the YourTTS model, a multilingual zero-shot multi-speaker text-to-speech model. Although the focus of the model is on TTS it can also do zero-shot voice conversion. For example, it was able to produce female voices even without being trained on female voices. Here, we fine-tuned the YourTTS model in English, pt-BR and ru-RU.


# Noise-robust Speech Recognition with 10 Minutes Unparalleled In-domain Data
We propose SimuGAN to simulate noisy spectrum from the clean spectrum.
In Simu-GAN, the generator maps the clean spectrum to the noisy spectrum and the discriminator distinguishes the simulated noisy spectrum from the real noisy spectrum. We also apply the multi-layer patch-wise contrastive loss to the generator (fig. 1). After SimuGAN training, only the generator is required to generate the simulated noisy data.
To train Simu-GAN, we utilize small amounts of clean/noisy data from the channel A from the RATS dataset.
We also propose a dual-path ASR system to improve the robustness of the ASR systems under noisy conditions. Specifically, we input the noisy speech generated by the Simu-GAN model and the corresponding clean speech as dual-path inputs into the conformer-based ASR system (fig. 2). We then reduce a KL divergence-based consistency loss between the two decoder outputs, as well as ASR losses for both outputs.


# MAESTRO: Matched Speech Text Representations through Modality Matching
We present a novel self-supervised modality matching algorithm Maestro. It can effectively use small amounts of transcribed speech data to unify representations learnt from massive amounts of untranscribed speech and unspoken text. The model can transfer to diverse downstream tasks such as ASR and Speech Translation.
When learning from unspoken text, speech-text alignment information is unavailable. Therefore, Maestro uses durations predicted from a duration prediction model in a fashion similar to speech synthesis (fig. 1).
We establish a new SOTA on VoxPopuli multilingual ASR.


# Maestro-U: Leveraging joint speech-text representation learning for zero supervised speech ASR
Zero-supervised speech ASR means learning ASR without the availability of any (in-language) transcribed resources. Most prior research either learns models for phoneme recognition (implicitly assuming an additional model for phoneme to grapheme conversion), or assumes the availability of grapheme to phoneme (G2P) models for text augmentation, but the construction of such models require as much human efforts as speech transcription.
We assume the availability of unlabeled speech and text in 102 languages, and the availability of supervised speech in 52 of these languages. Given these resources, we attempt to improve end-to-end ASR quality on the remaining 50 zero-supervised-speech languages. For unseen writing systems, we convert graphemes (text) into a common representation that is shared across all languages.
Out baseline Maestro model fails to perform well on this task (CER of 54.2% averaged over 50 languages).
We propose several improvements to the Maestro model, namely, the use of language embeddings and adapters, and use of byte level text representations. It results in a final zero supervised speech average CER of 30.8% (fig. 2). The training data, architecture and evaluation is shown in fig. 1.


# Self-supervised learning with random-projection quantizer for speech recognition
We propose BERT-based Speech pre-Training with Random-projection Quantizer (BEST-RQ), a simple and effective self-supervised learning algorithm for speech recognition (fig. 1).
The algorithm masks speech signals and feeds them to the encoder, and the encoder learns to predict the masked region based on the unmasked speech signals where the learning targets are labels provided by a random-projection quantizer, which projects speech signals to a randomly initialized matrix, and finds a nearest vector in a randomly initialized codebook (neither the projection matrix nor the codebook is updated throughout the learning process).
On LibriSpeech the algorithm achieves similar results as previous work with non-streaming models, and provides better improvement with streaming models.
We study the relation between representation learning quality and the self-supervised learning quality, and demonstrate that the two objectives are not inherently aligned. Such an observation is central to our design. Our self-supervised learning algorithm eliminates the requirement of representation learning through applying a random-projection quantizer.
IMO, it is not clear what is meant under representation learning here, and why is it eliminated.


# FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech
We introduce FLEURS, the Few-shot Learning Evaluation of Universal Representations of Speech benchmark, to catalyze research in low-resource speech understanding.
FLEURS is suited for a variety of speech tasks including ASR, Speech-to-Text and Speech-to-Speech Translation, Speech LangID, and Multilingual Speech-to-Speech and Speech-to-Text Retrieval.
FLEURS contains n-way parallel speech and text in 102 languages with transcripts and strong quality control.
To collect data, we start with the dev and devtest sets from FLoRes-101 dataset for machine translation. It contains 2009 sentences extracted from English Wikipedia and these sentences have been translated in 101 languages by human translators.
For each sentence in the 102 languages we collected three recordings by three different native speakerss, imposing a balance in terms of sex ratio when possible. All recordings are kept as they-are, either from quiet or noisy environment.
Each recording is evaluated by additional workers, rejecting some recordings and leaving us between zero and three recordings per sentence in the final dataset. In the first version of the dataset, about 21.5% of the sentences are missing because none of the three recordings were validated.
Most other datasets are aligned at a document level with automatic segmentation and alignment for segments, but FLEURS initially contains short utterances.
For ASR baseline, we add two LSTM layer to fine-tune wav2vec-BERT, using a CTC loss. We do not include meta information of language identification labels in modeling, and there is no language model used for hypothesis scoring. Experiments show that fine-tuning from multimodal speech+text pre-training (mSLAM) is slightly worse than fine-tuning from speech-only pre-training (wav2vec-BERT). Most gains of multimodal pre-training are observed in groups with large amounts of unlabeled speech.


# XTREME-S: Evaluating Cross-lingual Speech Representations
We introduce XTREME-S, a new benchmark to evaluate universal cross-lingual speech representations. It covers 102 languages from 10+ language families, 3 different domains and includes 7 downstram tasks divided into 4 different task families: recognition, translation, classification and retrieval (fig. 1). Test sets are available in open-source and are not hidden to the public.
XTREME-S also includes FLEURS, a recently introduced general-purpose multilingual evaluation dataset.
The training sets of XTREME-S range from a few hours to a few hundred hours of labeled data per language. This is a few-shot setting suited for low-resource understanding.
For ASR, we use three datasets: Fleurs, MLS and VoxPopuli, which cover more than 100 languages. In Fleurs ASR, multilingual fine-tuning is used and "unit error rate" (characters, signs) of all languages is averaged. In MLS ASR, multilingual fine-tuning on all languages is also used. The first baseline is a 600M wav2vec-BERT model trained on 429k unlabeled data in 51 languages from VoxPopuli, MLS, CommonVoice and BABEL, similar to XLS-R. The second is the 600m parameter mSLAM model.


# DRAFT: A Novel Framework to Reduce Domain Shifting in Self-supervised Learning and Its Application to Children's ASR
Including target domain data might not be feasible at the pretraining stage.
We propose domain responsible adaptation and finetuning (DRAFT): a three-stage (pretraining, adaptation, and finetuning) training paradigm to reduce domain shifting in pretrained speech models through an additional adaptation stage (fig. 1).
In DRAFT, residual adapters (RAs) are placed between blocks in the transformer and are responsible for learning domain specific information during an additional adaptation stage. The additional adaptation stage trains the model with finetuning data and with the same SSL loss that was used in the pretraining stage. To prevent catastrophic forgetting of the learned knowledge from source domain data, only RA parameters are updated during the adaptation stage.
DRAFT is agnostic to the type of SSL method used and is evaluated with APC, Wav2vec2.0, and HuBERT.
When performing DRAFT on SSL-pretrained speech models (trained with adult speech data) for child ASR tasks, we obtain significant improvements over baselines without adaptation for both causal (pretrained with APC) and noncausal transformers (pretrained with Wav2vec2.0 or HuBERT).
There is another similar paper (another version?) called "Towards Better Domain Adaptation for Self-Supervised Models: A Case Study of Child ASR".


# UFO2: A unified pre-training framework for online and offline speech recognition
ASR systems are typically categorized in: 1) online mode (a.k.a. streaming), which is developed to emit each hypothesized word as quickly and accurately as possible when it is spoken, and 2) offline mode, which aims to accurately emit the complete hypotheses after processing a full utterance. Self-supervised pre-training is usually offline, i.e. each represented feature is conditioned on the full-context inputs. When online ASR model is build upon SSL pre-training, the performance might be hindered due to the mode inconsistency between the pre-training and fine-tuning.
We propose a Unified pre-training Framework for Online and Offline (UFO2) Automatic Speech Recognition.
UFO2 unifies the online and offline modes into a single model. We apply 4 strategies on the feature extraction and training objectives. The full-context MHSA extracts offline-mode features (conditioned on the complete utterance). Simultaneously, the dynamic-chunked MHSA mimics different latency ranges for online-mode learning. The online and offline representation models share all the encoder and quantizer weights. Stop gradient is operated to decouple the impact of the online-mode objectives to the quantizer. The online and offline objectives are aggregated.
The proposed UFO2 significantly enhances the performance compared to the baseline methods on the LibriSpeech dataset. However, the performance in online mode still underperforms the offline mode.


# Augmentation Invariant Discrete Representation for Generative Spoken Language Modeling
We consider a generative spoken language modeling task (GSLM, language modeling from audios recordings only). Such speech LMs usually operate over discrete units obtained from quantizing internal representations of self-supervised models.
We focus on measuring and improving the robustness of discrete input representations for GSLM, with respect to variations such as time-stretch, pitch-shift, additive-noise, and reverberation.
We propose Unit Edit Distance for evaluating the robustness.
We show the lack of robustness of GSLM models (such as HuBERT).
We propose a method for learning robust discrete representation on top of any speech SSL model. We forward a clean signal through an encoder followed by a pre-trained quantizer (k-means). Next, we forward an augmented signal through the same encoder, followed by a new quantizer. The CTC loss between the deduplicated output of the clean signal and the output of the augmented signal is used to learn the parameters of the new quantizer (fig. 3)
We evaluate the proposed method using the standard GSLM setup, i.e., ABX, sWUGGY, sBLIMP.


# Toward a realistic model of speech processing in the brain with self-supervised learning
We compare wav2vec 2.0 to the brain activity of 412 English, French, and Mandarin individuals, while they listened to approximately one hour of audio books.
To quantify the similarity between the network’s activations X and the brain recordings Y, linear mapping is fitted to predict the brain response Y given X.
We show that wav2vec 2.0 learns brain-like representations with as little as 600 hours of unlabelled speech – a quantity comparable to what infants can be exposed to during language acquisition. Low-level brain areas (A1, A2) are best predicted by the first transformer layers, higher level areas (IFG, STS) are best predicted by deeper layers. Remarkably, this hierarchy extends to supplementary motor and motor areas in both hemispheres.
The auditory-, speech-, and language-specific representations learned by the model converge to those of the human brain.
While our analyses suggest that learning allows wav2vec 2.0 to capture some lexical features in its deep layers, it remains unclear whether these layers also capture complex syntactic structures, such as recursive syntactic trees.


# Are discrete units necessary for Spoken Language Modeling?
Spoken language modeling usually relies on transforming the audio into a sequence of discrete units and then training a language model directly on such pseudo-text. Is such a discrete bottleneck necessary (fig. 1)?
We show experimentally that discretization is beneficial for spoken language modeling. We show that discretization disentangles linguistic information from non-linguistic signals, forcing the transformer to focus on linguistic ones. But we can get rid of discrete bottlenecks by using low-level continuous inputs so long as we still use discrete targets. When there are no discrete units at all, the LM performances are still limited, even if using a NCE loss could help a bit with syntactic and semantic metrics.
On the basis of this study, we train a language model on the discrete units of the HuBERT features, reaching new SOTA results in the lexical, syntactic and semantic metrics of the Zero Resource Speech Challenge 2021.


# Comparative layer-wise analysis of self-supervised speech models
The authors extends the previous work "Layer-wise Analysis of a Self-supervised Speech Representation Model" that focused on wav2vec 2.0 to 11 pre-trained models. All the models in this work use a masking-based pretext task, thus using both left and right context to recover the masked segment (target).
As in our previous work, we use projection-weighted canonical correlation analysis (PWCCA) that returns a correlation-based measure given N pairs of vectors. Using PWCCA, we measure the similarity between layer representations and various variables of interest: mel filter bank features (acoustic information), one-hot encoded phone labels, and one-hot encoded word labels.
CCA similarity with local features (CNN outputs) are shown in fig. 1. Some models have a clear autoencoderstyle pattern, i.e., high similarity with the input for the initial and final layers and a drop in similarity for the middle layers. This behavior is most prominent in models trained to recover (in some sense) the local features.
Fig. 2, 3, 4 show the layerwise spectrogram, phonetic and word similarity respectively. Models that have a strong autoencoder-style dynamic (W2V2, XLSR-53, and FaST-VGS+) tend to have a peak in both phonetic and word content in one or more of the intermediate layers. hese models have the same masking-based contrastive loss that recovers the local features. For the models which are trained to predict discrete units learned in an intermediate layer (HuBERT, WavLM, AV-HuBERT) the phonetic and word information appears to be concentrated toward higher layers.
We also measure the performance of individual layers on downstream tasks, using a prediction model on top of the frozen representations, trained on labeled data for the task. Do layers with high property content also perform better on downstream tasks that are expected to benefit from that property? PR and ASR performance are well-correlated with both CCA-phone and CCA-word scores. Semantic task (SLURP-action, SLURP-scenario, and IC) performance is much more correlated with CCA-word than with CCA-phone. The best-performing layer isalways lower than at least the top two layers and is close to the layers observed to have the most phonetic and word-level content as measured by CCA.
Evaluating each individual layer to find the best-performing layer is much more computationally demanding than evaluating CCA. Our findings suggest using CCA-word/CCAphone to narrow down the choice of layers.
We compare the best layer performance to the task performance using a learnable weighted sum of all layers . In general, they perform on par (fig. 7). It is natural to ask whether the layer weights learned in all-layers experiments are themselves a good indicator of usefulness for downstream tasks. We find that the mean rank correlation between layer weights and task performance is much lower than the mean rank correlation between CCA-word and task performance. So, layer weights are relatively unreliable.


# Adaptive multilingual speech recognition with pretrained models
We show that using multilingual pretrained acoustic (wav2vec 2.0) and language (MBART50) models for the encoder and decoder respective brings a large improvement for seq2seq models for ASR. Encoder pretraining is more impactful for languages with higher resources while the decoder counterpart is more effective for languages with medium-low resources. Surprisingly, many languages with extremely low resource (less than 5 hours) do not benefit much from this combination.
The language specific modulation techniques such as language adapters and factorized adaptive weights have a strong impact on especially the low-resource languages mentioned above.
We enhance the encoder with either content bias and position bias to self-attention, or stacking MBART50 encoder on top of the wav2vec encoder (without any length conversion) and achieve more improvements in WER. We hypothesize that the latter helps the cross-attention layers because they are familiar with the output of the text encoder during training.


# Robust Speech Recognition via Large-Scale Weak Supervision
Current SSL audio encoders such as Wav2Vec 2.0 lack an equivalently performant decoder mapping those representations to usable outputs, necessitating a finetuning stage in order to actually perform a task such as speech recognition. An additional risk with fine-tuning is learning brittle and spurious patterns that don’t generalize to other datasets.
As demonstrated earlier, multi-domain pre-training yields higher robustness and generalize much more effectively to held-out datasets than models trained on a single source. However, there is still only a moderate amount of this data easily available.
We release Whisper, an encoder-decoder Transformer for speech recognition, with a small stem of two convolution layers, and byte-level BPE text tokenizer (fig. 1), trained on many different tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection.
Whisper was trained on 680,000 hours of weakly supervised audio data (including 117,000 hours or recordings on 96 languages other than English), splitted into 30-second segments. We construct the dataset from audio that is paired with transcripts on the Internet. This results in a very diverse dataset covering a broad distribution of audio from many different environments, recording setups, speakers, and languages. When collecting dataset, we developed several automated filtering methods to improve transcript quality.
Many transcripts on the internet are not actually human-generated but the output of existing ASR systems. It was shown that training on mixed human and machine-generated data can impair the performance of translation systems. We developed many heuristics to detect and remove machine-generated transcripts from the training dataset. For example, an all-uppercase or all-lowercase transcript is very unlikely to be human generated.
For an additional filtering pass, after training an initial model we manually inspect data sources with high WER. This showed a large amount of only partially transcribed or poorly aligned/misaligned transcripts as well as remaining low-quality machine-generated captions that filtering heuristics did not detect.
We train on all audio, including segments where there is no speech.
We use multi-tasking with a simple format to specify a task and conditioning information as a sequence of input tokens to the decoder. First, we predict the language being spoken which is represented by a unique token for each language in our training set (99 total), or <|nospeech|> token. The next token specifies the task with an <|transcribe|> or <|translate|> token. After this, we specify whether to predict timestamps or not by including a <|notimestamps|> token. At this point, the task and desired format is fully specified, and the output begins.
With some probability we also condition on the history of text of the transcript (in the hope that it will learn to use longer-range text context to resolve ambiguous audio). We add the transcript text preceding the current audio segment to the decoder’s context. We indicate the beginning of prediction with a <|startoftranscript|> token. We only mask out the training loss over the previous context text, and train the model to predict all other tokens.
During early development we observed that Whisper transcribes plausible but almost always incorrect guesses for the names of speakers. This happens because many transcripts in the pre-training dataset include the name of the person who is speaking. To avoid this, we fine-tune Whisper models briefly on the subset of transcripts that do not include speaker annotations which removes this behavior.
The models are trained for 2^20 updates which is between two and three passes over the dataset. Due to low number of epochs, we do not use any data augmentation or regularization. After the original release of Whisper, we trained an additional Large model (denoted V2) for 2.5X more epochs while adding SpecAugment, Stochastic Depth and BPE Dropout.
Systems that output transcripts that would be judged as correct by humans can still have a large WER due to innocuous differences in transcript style. This is particulary important for zero-shot evaluation, when the model do not observe any examples of specific datasets transcript formats. For several datasets, we observe WER drops of up to 50 percent usually due to a quirk such as a dataset’s reference transcripts seperating contractions from words with whitespace. We opt to address this problem with extensive standardization of text before the WER calculation. Our text normalizer was developed through iterative manual inspection. We caution this development procedure comes at a risk of overfitting to the transcription style of Whisper models. We are releasing the code for our text normalizer.
Following the paper "Measuring Robustness to Natural Distribution Shifts in Image Classification", we examine 1) overall robustness (average performance across many datasets; here we use a suite of 12 other academic speech recognition datasets), and 2) effective robustness (the difference in performance between the reference dataset and out-of-distribution dataset; we use LibriSpeech as reference dataset due to its central role in modern ASR).
We evaluate Whisper in a zero-shot setting without using any of the training data from test distributions.
We compare Whisper models with both human performance and standard fine-tuned machine learning models.
Even the smallest zero-shot Whisper model, which has only 39 million parameters and a 6.7 WER on LibriSpeech test-clean, is roughly competitive with the best supervised LibriSpeech model when evaluated on other datasets (fig. 2, table 2). The best zero-shot Whisper models roughly match human accuracy and robustness.
So, our key finding is that multi-domain training increases robustness. This finding has been replicated across many fields in addition to speech recognition including NLP and CV.
As for multi-lingual speech recognition, Whisper performs well on Multilingual LibriSpeech (MLS) in a zero-shot setting. We do use a simple text standardizer for this result which prevents direct comparison or claims of SOTA performance. On VoxPopuli, however, Whisper significantly underperforms prior work (while in prior work models were trained on VoxPopuli data, which is 10 times larger than MLS). However, these two benchmarks are somewhat narrow since they only include 15 unique languages, so we also report performance on the Fleurs dataset. We find a strong squared correlation coefficient of 0.83 between the log of the WER and the log of the amount of training data per language (fig. 3): WER halves for every 16× increase in training data. WER also can be poor when BPE tokenizer is a poor match for some languages.
We study the translation capabilities of Whisper models by measuring their performance on the X→en subset of CoVoST2. We achieve a new state of the art of 29.1 BLEU zero-shot without using any of the CoVoST2 training data.
We also evaluate Whisper on language identification.
We also measure robustness to Additive Noise on LibriSpeech dataset. When either white noise or pub noise from the Audio Degradation Toolbox (Mauch & Ewert, 2013) was added to the audio. All other (LibriSpeech-trained) models quickly degrade as the noise becomes more intensive, performing worse than the Whisper model under additive pub noise of signal-to-noise ratio (SNR) below 10 dB. his showcases Whisper’s robustness to noise.
Whisper models are trained on 30-second audio chunks and cannot consume longer audio inputs at once, which presents challenges in real-world applications. We developed Long-form Transcription: a strategy to perform buffered transcription of long audio by consecutively transcribing 30-second segments of audio and shifting the window according to the timestamps predicted by the model. It is crucial to have beam search and specific temperature scheduling for this to perform good. We use beam search with 5 beams using the log probability as the score function, to reduce repetition looping which happens more frequently in greedy decoding. We start with temperature 0, i.e. always selecting the tokens with the highest probability, and increase the temperature by 0.2 up to 1.0 when either the average log probability over the generated tokens is lower than −1 or the generated text has a gzip compression rate higher than 2.4.
We also specifically compare with human performance. Whisper’s English ASR performance is not perfect but very close to human-level accuracy.
We also trained a series of medium-sized models on subsampled versions of the dataset which are 0.5%, 1%, 2%, 4%, and 8% of the full dataset size, with early stopping based on the validation loss, and taking EMA of the parameters. We see significant variability in improvement rates across tasks and sizes. For English ASR, fter 50K hours the diminishing returns observed, that could be explained by saturation effects when approaching human-level performance. For X -> en translation, performance is practically zero when training on 7,000 hours of audio or less and then follows a roughly log-linear improvement trend till 54,000 hours before also showing diminishing returns. Overall results could suggest that the current best Whisper models are under-trained relative to dataset size and performance could be further improved by a combination of longer training and larger models. Also, we are nearing the end of performance improvements from dataset size scaling for speech recognition.
We also compared the performance of models trained on just English ASR with our standard multitask and multilingual training setup. We adjust for the amount of FLOPs spent training on the task of English ASR as only 65% of compute is spent on this task in a joint training setup. For small models trained with moderate amounts of compute, there is indeed negative transfer between tasks and languages (that is, multitasking harms). However, for our largest experiments  multitask and multilingual models outperform their English-only counterparts, demonstrating positive transfer from other tasks.
Many remaining errors, particularly in long-form transcription, includes problems such as getting stuck in repeat loops, not transcribing the first or last few words of an audio segment, or complete hallucination where the model will output a transcript entirely unrelated to the actual audio. We suspect fine-tuning Whisper models on a high-quality supervised dataset and/or using reinforcement learning to more directly optimize for decoding performance could help further reduce these errors.
Another clear route for improvement is increasing the amount of data for rarer languages.
While we studied only zero-shot transfer performance, it is likely that results can be improved further by fine-tuning. It also allows for direct comparisons with prior work since it is a much more common evaluation setting.
It’s currently unclear to what degree the benefits of Whisper stem from training its encoder, decoder, or both. This could be studied by either ablating various design components of Whisper, such as training a decoder-less CTC model, or by studying how the performance of existing speech recognition encoders such as wav2vec 2.0 change when used together with a language model. It is also possible that the results could be further improved by incorporating unsupervised pre-training.


# Massively Multilingual ASR on 70 Languages: Tokenization, Architecture, and Generalization Capabilities
A problem in multilingual ASR: when we scale up the number of languages, vocabulary size grows.
We explore large-scale multilingual on 70 languages with 150,000 hours dataset (fig. 2).
We use an end-to-end Transducer model that is composed by encoder (CNN + Transformer), predictor, and joiner modules. We investigate two types of Transducer models: (1) shared input embedding and output architecture, (2) language-specific multiple input embedding, and output linear architecture (fig. 1). We also try subword tokenization.
We evaluate our model with test data on two different domains for every language: vid-clean and vid-noisy (more acoustically challenging).
Shared character strategy shows inferior results compared to language-specific inputs and outputs.
One advantage of this language-specific architecture is that we could represent the same token between different languages with different embedding and weight matrices. Thus, we could disambiguate characters and subwords that look the same in the written space but sound different.
Shared character tokenization lead to subpar performance, maybe due to the high variation of decoding timestep between different languages. We show that by minimizing the variance of decoding steps between languages with clever combinations between subwords and characters, we could significantly improve our multilingual result.
Our multilingual model could generalized well on the new dataset (MLS). We achieve 9.5% WER on zero-shot and 7.5% WER after finetuning, that is competitive with SOTA performance.
We plan to scale up the amount of training data by adding pseudo-labeling pipeline in every language.


# ASR in German: A Detailed Error Analysis
In ASR, WER or CER metrics do not provide any insight into the nature or impact of the errors.
We evaluate ASR models pretrained on the German language on diverse test datasets.
Conformer Transducer outperforms all other models (including Wav2Vec 2.0, Conformer CTC etc.) regarding WER.
The lowest difference between average and median (over datasets) we count as an indicator for robustness.
Wav2Vec 2.0 does exceptionally bad for German TED and ALC. We hypothesize that this stems from filling words like “äh” and “hm” occurring in their output transcript predictions, which, in contrast, are omitted in the predictions of all other models. These fillers are also missing in all ground truth transcripts.
1. Negligible Errors (9%). These are different forms of otherwise correct transcript predictions, like nonnormalized abbreviations such as “et cetera” or “etc.”
2. Minor Errors (noncontext-breaking) (12%). Models which were trained on less German data often produce transcript predictions with redundant letters, omit single letters or predict hard instead of soft vowels and vice versa (e.g., confusion between d and t) without distorting the meaning. These spelling errors can usually be corrected when utilizing a language model. While increasing CER only slightly, WER quickly rises to unrealistic values if these minor errors are included.
3. Major Errors (context-breaking) (19%). These are fully incorrectly transcribed or omitted words and omitted or inserted letters that change the meaning of a transcript or exclude necessary information. These errors were further divided into subgroups and their causes traced back to systematic errors within the training data.
3.1. Naive Normalization, such as years and large integers: "One Thousand Nine Hundred Sixty-Three" instead of "Nineteen Sixty-Three", and pronounced punctuation marks such as "comma". Here models trained on one type of dataset will generate errors when evaluated with the other type of dataset.
3.2. Various problems within datasets, such as pronounced "weil" (because) is transcribed in the ground truth as "denn" (since), or certain terms such as "paragraph" (article) were transcribed as "ziffer" (subparagraph) within the ground truth transcripts. As a result, models trained on this wrong data consistently predicted these spoken words. Additionally, for poorly edited audio inputs (starting in the middle of a sentence or word), models predicted sentence beginnings that were not present in the audio, and in most cases turned out to be incorrect. This type of error fits into the category "hallucinations".
4. Names, Loan Words and Anglicisms (20%) are a commonly occurring types of errors, since names can often be spelled in several ways and only a small fraction of common names is usually found in training datasets. Foreign words are strongly domain dependent, and some anglicisms have homophonic pronunciations to German phonemes. Anglicisms are closely related to code-switching ASR.
5. Homophones (3%) occur when contextual information is lost, but the underlying phonemes were in general correctly interpreted by the system, e.g., “Graph” and “Graf” (a noble title).
6. Flawed Ground Truth Transcripts (18%). These cases may be caused by incorrect normalization of numbers and symbols, and strong deviations in the alignment process during the automated creation of data sets.
7. Ambiguous Audio Input (11%), when certain words are be pronounced or perceived indistinctly. Here even human listeners were unable to produce clear and correct transcripts from audio recordings, because the pronunciation was too unclear. This could be traced back to speakers having learned German as a second language, multiple German dialects, simple pronunciation errors and slips of the tongue. Increasing the robustness of ASR systems for non-native speakers and dialects is an ongoing research topic.
8. Flawed Audio Input (8%) with cutoffs of spoken words at the beginning or end of audio snippets.
We propose sevaral solutions: verification of (correct) normalization, extension of vocabulary through text to speech, training on phoneme vocabulary, audio preprocessing.


# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data
We propose SpeechLM, a cross-modal Speech and Language Model to explicitly malign speech and text pre-training with a predefined unified discrete representation. We introduce two alternative discrete tokenizers to bridge the speech and text modalities, including phoneme-unit and hidden-unit tokenizers, which can be trained using a small amount of paired speech-text data.
We propose two pre-training tasks. One is Unit-based Masked Language Modeling (UMLM) trying to predict the unit tokens from the masked speech. The other one is Unit-based Connectionist Temporal Classification (UCTC) task, aiming at reconstructing the whole text sequences from the masked unit sequences. To better align the representations of speech and text, we also adopt a Random Swapping Mechanism for the UMLM task.
SpeechLM enhanced by textual data significantly outperforms its speech-only counterparts on various spoken language tasks, e.g., ASR, speech translation (ST), and universal representation evaluation framework SUPERB.


# SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training
For the cross-modal speech-to-text models, a key problem is how to naturally connect the speech encoder and the text decoder. An intermediate hidden-unit representation (such as one in HuBERT) can be the bridge between modalities.
We propose a unified speech-unit-text pre-training method (SpeechUT) that decouples the speech-to-text model into speech-to-unit and unit-to-text models (fig. 2), to take advantage of a large amount of unpaired speech and text data for pre-training.
SpeechUT connects the speech encoder and the text decoder by the unit encoder.
In the pre-training stage, SpeechUT performs multi-task pre-training with the following tasks. The first is the speech-to-unit objective similar to HuBERT, where the model needs to predict the unit of the masked positions based on the non-mask regions in a speech sequence. The second is the unit-to-text task is performed as a regular encoder-decoder based sequence-to-sequence task, conditioned on the output of the unit encoder; we also formulate a joint CTC objective which directly predicts the target text sequence from the unit encoder. Note that in S2U and U2T tasks, the unit serves as the target and the input, respectively. Finally, to enhance the unit-in, unit-out property, SpeechUT performs an additional masked unit modeling (MUM) task, with the training data combining all the units in S2U and U2T tasks. To better align the speech and unit representations in the unit encoder, we adopt a simple embedding mixing mechanism for S2U task, which is to mix the embeddings of two modalities in one sequence.
To obtain training data, we need to construct the speech-unit paired data, and the unit-text paired data. We introduce the speech-to-unit (S2U) and text-to-unit (T2U) offline generators. SpeechUT employs a small amount of paired ASR data to train the T2U generator.
After pre-training, we drop the unit pre-net and stack the speech encoder, the unit encoder and the text decoder into a complete sequence-to-sequence model, which can be fine-tuned for any speech-to-text task, such as ASR and ST.
According to our ablation studies, the embedding mixing mechanism has the biggest impact, and the CTC loss, as a part of the U2T task, has a minor influence on the fine-tuning performance. The MUM loss has the minimum effect, so we speculate that the U2T task has already modeled the unit well.
We acheieve SOTA on speech recognition and speech translation tasks.


# A Noise-Robust Self-supervised Pre-training Model Based Speech Representation Learning for Automatic Speech Recognition
We observe that wav2vec2.0 pre-trained on noisy data brings a performance degradation on the clean test set.
We propose a modified pre-training scheme: the noisy speech and clean speech are sent into a shared feature encoder, and the noisy feature is input to the transformer encoder, while the clean feature is fed to the vector-quantization (VQ) module, which provides clean training targets for the transformer encoder.
To synthetize noisy data, we randomly select noise samples and mix with the clean speech at different SNRs.
The resulting model achieves a much better performance on noisy data at the cost of a tiny performance sacrifice on the clean test set.


# WhisperX: Time-Accurate Speech Transcription of Long-Form Audio
We propose WhisperX, a system for efficient speech transcription of long-form audio with accurate word-level timestamps. It consists of 3 additional stages to Whisper transcription: (i) pre-segmenting the input audio with an external Voice Activity Detection (VAD) model; (ii) cut and merging the resulting VAD segments into approximately 30 seconds input chunks with boundaries lying on minimally active speech regions enabling batched whisper transcription; and finally (iii) forced alignment with an external phoneme model to provide accurate word-level timestamps.
We demonstrate SOTA performance on long-form transcription and word segmentation benchmarks.
Pre-segmenting audio enables a 12x transcription speedup via batched inference.


# Using fine-tuning and min lookahead beam search to improve Whisper
We compare the following fine-tuning strategies for Whisper-Tiny on Vietnamese low-resource language: 1) full-paremeter, 2) full-paremeter with decoupling input an output embedding layers, 3) fine-tuning decoder only, 4) fine-tuning decoder embedding layer only, 5) fine-tuning decoder embedding layer only with decoupling input an output embedding layers, 6) LoRA with decoupling input an output embedding layers and different hyperparameters.
We fine-tune on our collected dataset and test on FLEURS and CommonVoice 9 (for Vietnamese?)
Fine-tuning both the audio encoder and text decoder maximises performance. Applying high-rank LoRA leads to the greatest model improvement, with less than half the number of trainable parameters compared to full-parameter fine-tuning with decoupling embedding. Decoupling input an output embedding layers also improves performance.
We suggest Filter-Ends and Min Lookahead as improvements to Whisper’s decoding algorithm. We prove that Min Lookahead is expected to outperform standard beam search.


# Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling
We leverage pseudo-labelling to distill the Whisper model into a smaller variant, called Distil-Whisper. The encoder is entirely copied from the teacher to the student and frozen during training. The student’s decoder consists of only two decoder layers, which are initialised from the first and last decoder layer of the teacher, so the distilled model is 5.8 times faster with 51% fewer parameters. The model is trained on a weighted sum of 1) KL divergence with teacher output distribution, and 2) next token prediction of pseudo labels.
We found there to be little difference in the downstream performance of the distilled model after pseudo-labelling using either greedy or beam-search, and so we opted to pseudo-label the training data with greedy decoding for its faster inference speed.
By sharing the same encoder weights as Whisper, Distil-Whisper is designed to be paired with Whisper for speculative decoding, yielding a 2 times speed-up while ensuring the same outputs as the original model.
Distil-Whisper maintains the robustness of the Whisper model, performing to within 1% WER on OOD test data in a zero-shot transfer setting.
On long-form evaluation, the distilled model outperforms Whisper by 0.1% WER. We show that this performance gain is due to a lower propensity to hallucinate than the original Whisper model. Hallucinations are characterised by either the repetitive generation of identical sequences, or predicting passages of text not spoken in the audio input and are most prevalent in long-form audio transcription, particularly when the audio contains large amounts of silence between spoken utterances. To quantify the amount of repetition and hallucination in the predicted transcriptions, we measure the number of repeated 5-gram word duplicates (5-Dup.) and the insertion error rate (IER) over the four OOD long-form datasets.
IMO, the question: does KL loss already includes Pseudo Labeling loss, if the latter is obtained with greedy decoding?


# Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations
We propose Miipher (multiple features integrated speech restoration), a robust speech restoration (SR) model, that convers degraded speech signals into high-quality ones. The enhanced speech can be further used to train text-to-speech (TTS) models, because the quality of speech generation is directly affected by that of the training samples.
We especially focus on the following two difficult degradations where SR frequently fails: phoneme masking by noise and/or reverberation, and phoneme deletion due to codecs and/or down-sampling. If a noisy sample lacks a phoneme, it introduces an unrecoverable error. To solve these issues, we 1) use a speech representation extracted from w2v-BERT, instead of a log-mel spectrogram, 2) we consider the deleted phoneme reconstruction problem as a text-conditioned speech inpainting, and use a text representation extracted from PnG-BERT (fig. 1).
Since SSL features often lose speaker information, we also use a speaker embedding extracted from audio with a streaming Conformer-based speaker encoding model. The model was trained on the dataset described in "Parameter-free attentive scoring for speaker verification" while minimizing the generalized end-to-end extended-set softmax (GE2E-XS) loss. Speaker embedding is combined to PnG-BERT features using a CNN-based simple feature-wise linear modulation (FiLM) layer.
Then a DF-Conformer-based feature cleaner predicts clean w2v-BERT features. We also apply the 5-layer CNN Post-Net proposed in Tacotron2. We iterate twice the entire feature cleaning process consisting of the feature cleaner and the Post-Net, where the parameters of the layers are shared. We used a combined loss function of MAE, MSE, and a spectral convergence loss. This loss is calculated before and after the Post-Net, and calculated for both iterations.
Then, a WaveFit neural vocoder synthesizes a restored waveform from cleaned w2v-BERT features. In addition to the original adversarial loss function proposed in WaveFit, we used the multi-period discriminator (MPD).
We trained the proposed model with a proprietary dataset that contains 2,680 hours of noisy and studio-quality speech pairs. To apply the noise, we used the TAU Urban Audio-Visual Scenes 2021 dataset, internally collected noise snippets that simulate conditions like cafe, kitchen, and cars, and noise and music sources. The noisy utterances were generated by mixing randomly selected speech and noise samples from these datasets with signal-to-noise ratio (SNR) from 5 dB to 30 dB. In addition, we augmented the noisy dataset with 4 patterns depending on the presence or absence of reverberation and codec artifacts.


# Some voices are too common: Building fair speech recognition systems using the Common Voice dataset
We use the French Common Voice dataset to quantify the biases of a pre-trained wav2vec 2.0 model toward several demographic groups.
We fine-tune the pre-trained model on a variety of fixed-size, carefully crafted datasets. Results highlights the importance of prioritizing speaker diversity over dataset size and demographic diversity when collecting audio data.


# Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data
We present an Open Whisper-style Speech Model (OWSM), which reproduces Whisperstyle training using an open-source toolkit and publicly available data. We will provide reproducible recipes encompassing the entire pipeline, including data preparation, training, inference, and scoring. Furthermore, we will release pre-trained models and training logs.
Our multitask data format mostly follows OpenAI Whisper (fig. 1). Our model is designed to support any-to-any speech-to-text translation, whereas Whisper can only perform any-to-English translation.
OWSM additionally employs a joint CTC loss for ASR targets. In our preliminary experiments, we observed suboptimal convergence of the attention-based encoder-decoder, and incorporating a joint ASR CTC loss to the encoder output can stabilize training and expedite convergence.
We combine training sets from various publicly available ASR and ST corpora. Our largest dataset comprises 180k hours of labeled audio data (approximately one quarter of the Whisper's total data). We have developed new data preparation scripts tailored specifically for Whisper-style training (long-form audio data).
Our training data is gathered from many public corpora with inconsistent case and punctuation. During inference, we find that OWSM models are able to recognize the corpus and generate outputs that are consistent with the training data format. In the future, we will normalize the text to address this issue. Note that this analysis demonstrates the benefit of using public data and open-source code, without which we cannot discover such issues.
For inference, OpenAI Whisper implements both greedy decoding and beam search with temperature fallback. The latter is a complicated procedure relying on many heuristics. Our OWSM utilizes the ESPnet framework, ensuring compatibility with various decoding algorithms including greedy search, beam search, and joint CTC/attention decoding (for ASR only).
The current OWSM still falls behind Whisper in many benchmarks.
IMO, the authors evaluate on the datasets which are in-domain for OWSM and out-of-domain for Whisper.


# Emphasizing Unseen Words: New Vocabulary Acquisition for End-to-End Speech Recognition
ASR systems need to continuously acquire new vocabulary.
We generate out-of-vocabulary (OOV) words using text-to-speech systems. We choose 100 OOV words appearing in LRS3-TED dataset but not existing in LibriSpeech dataset. Then, we crawl texts including the new words from the Internet and synthesize audio with TTS systems. Note that the same OOV words (in another itterances) are used to test the model.
We enlarge the classification loss on utterances containing OOV words (sentence-level), or rescale the gradient used for back-propagation for OOV words (word-level), when fine-tuning a previously trained model on synthetic audio. To overcome catastrophic forgetting, we employ L2 regularization and elastic weight consolidation (EWC). This can support continual learning of an ASR system.
We use two-pass hybrid CTC/attention ASR architecture (fig. 5).
Compared with previous methods that just fine-tune synthetic audio with EWC, the experimental results on the LibriSpeech benchmark reveal that our proposed loss rescaling approach can achieve significant improvement on the recall rate with only a slight decrease on WER.
Word-level rescaling is more stable than utterance-level rescaling and leads to higher recall rates and precision on OOV word recognition.


# Bigger is not Always Better: The Effect of Context Size on Speech Pre-Training
We experiment on ABX phone discriminability task: ic scores the number of times model representations of tokens of a given phoneme are more similar to other representations of the same versus another phoneme.
We find that phone discriminability in the pre-trained CPC model representations peaks at around 40 ms of preceding context, and that having too much context (beyond around 320 ms) substantially degrades the quality of the representations.
This pattern also transfers to supervised ASR on top of frozen pre-trained representations.
So, we saw strong support for the hypothesis that too much context is detrimental. Because the downstream ASR model can perform significant pattern recognition on its own, it may be more important for the upstream model to retain a high-fidelity representation of the input than to perform extensive pre-processing.


# AudioPaLM: A Large Language Model That Can Speak and Listen
We introduce AudioPaLM, a unified speech-text LLM, capable of consuming and producing both speech and text.
AudioPaLM fuses text-based (PaLM-2) and speech-based (AudioLM) language models.
AudioPaLM uses joint vocabulary that can represent speech and text with a limited number of discrete tokens. This allows training a decoder-only model on a mixture of tasks that involve arbitrarily interleaved speech and text (speech recognition, text-to-speech synthesis, speech-to-speech translation). We initialize decoder weights with those of a text-only LLM.
We achieve SOTA on AST and S2ST benchmarks, and competitive performance on ASR benchmarks.
AudioPaLM performs zero-shot AST with speech input/target language combinations that were not seen in training.
AudioPaLM is able to transfer a voice across languages based on a short spoken prompt.


# Improving Fairness and Robustness in End-to-End Speech Recognition through unsupervised clustering
We focus on solving fairness and robustness issue in ASR while meeting privacy requirement (regulators have passed down strict laws that prohibit the use of demographic and other Personal & Private Information in building AI systems).
We propose clustering the training data using utterance level embeddings extracted with a speaker ID model and usethe resulting cluster ID as an additional feature in training instead. At inference time, we give each utterance an “unknown” cluster ID as an additional feature.


# From English to More Languages: Parameter-Efficient Model Reprogramming for Cross-Lingual Speech Recognition
The success of current neural ASR models is still related to the scale of training data.
We propose Conformer-based ASR Reprogramming (CAR) for cross-lingual adaptation, that makes the model frozen and only inserts few trainable modules.
CAR is based on The Neural reprogram method (see "Adversarial Reprogramming of Neural Networks") which mainly adds trainable parameters at its input level of a pre-trained model.
Our neural reprogramming has three major components (fig. 1): (1) input reprogramming, associated with standard model reprogramming or input-prompting, (2) latent space reprogramming, related to the residual adapters, and (3) multilingual graphemes pre-training which aims to resolve the existing challenges of cross-lingual learning on graphemes mismatching.
We only require 11M (4.8% of its full pretrained model) trainable parameters to achieve 11.9% WER cross seven languages in MLS benchmark for ASR task.


# Automatic Data Augmentation for Domain Adapted Fine-Tuning of Self-Supervised Speech Representations
We propose a novel supervised domain adaptation method for self-supervised speech representations. We apply properly calibrated data augmentations on a large clean dataset, bringing it closer to the target domain, and using it as part of an initial fine-tuning stage. Augmentations are automatically selected through the minimization of a conditional-dependence estimator, based on the target dataset (fig. 1). After this, fine-tuning on the small target domain dataset is performed.
The method is validated with an oracle simulated experiment and experiments with naturally noisy datasets.


# Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages
We introduce the Universal Speech Model (USM) that performs ASR across 100+ languages.
Our training pipeline utilizes three types of datasets: Unpaired Audio (including 12M hours of YouTube-based audio covering over 300 languages), Unpaired Text and Paired ASR Data (including 90k hours of labeled multilingual data covering 73 language and 100k hours of en-US pseudo-labeled data generated by noisy student training).
Our model is 2B-parameter Conformer.
The first step (fig. 1) is Unsupervised Pre-training: BEST-RQ (BERT-based Speech pre-Training with Random-projection Quantizer). We find that BEST-RQ pre-training can effectively scale to the very large data regime with a 2B Conformer, comparing favorably against Wav2Vec 2.0 and W2v-BERT in this setting.
The second step is MOST (Multi-Objective Supervised pre-Training). Here, we optimize a weighted sum of the BEST-RQ masked language model loss, along with the text-injection losses (including the supervised ASR loss and modality matching losses).
The third step is Supervised ASR Training of ASR models trained with CTC or RNN-T decoder (these models have been observed to hallucinate compared to attention-based seq-to-seq decoders). We also introduce chunk-wise attention (fig. 4): the USM-CTC/LAS models trained with it is able to produce high-quality transcripts for very long utterances.
We compare the performance of our models against public baselines, including Whisper large-v2. For the massively multilingual speech recognition test dataset from YouTube, we observe that Whisper hallucinates in many languages, resulting in a WER exceeding 100%. We exclude languages for which Whisper produces > 40% WER, and also use segmented decoding for Whisper with 30-second segments to further reduce the effect of hallucinations. Still, our USM-LAS and USM-CTC models outperform Whisper by a wide margin on YouTube en-US, despite training on significantly less supervised data.
We investigate whether MOST representations are useful for adapting the model to new domains by freezing the entire learned encoder produced by MOST and adjusting a small amount of parameters added to the network by residual adapters.By adding only 2% to the total number of parameters, the MOST only performs slightly worse than the fine-tuning baselines. The small number of parameters being trained in this approach makes it feasible to extend our system to a large number of new domains and new tasks, even with a limited amount of training data, such as in FLEURS.


# Whispy: Adapting STT Whisper Models to Real-Time Environments
We introduce Whispy, a production-ready system that can be used as a self-contained transcription service (fig. 1). Whispy is a wrapper around the Whisper pretrained models. The overall system lives within an HTTP server.
Whispy processes short audio chunks that accumulate within a shifting buffer. An agreement algorithm based on the Levenshtein distance extracts the most accurate transcript suggestions when overlapping portions of the buffer are transcribed.
The produced transcription is filtered to detect potential hallucinations, that is required because of the intrinsic tendency of large text-generation models to produce, from time to time, unreliable or unpredictable outputs. Other than content-oriented hallucinations, Whisper can produce hallucinatory artifacts resulting in the repetition of a single token or sequence of tokens. We designed a naive hallucination filter capable of reliably detecting repeating tokens and skim them. However, during our test campaigns we did not experience a sufficiently large number of hallucinatory events to determine the goodness of the hallucination filter.
Whispy maintains transcription performance comparable to its offline counterparts, while exhibiting minimal latency.


# Hallucinations in Neural Automatic Speech Recognition: Identifying Errors and Hallucinatory Models
The main difference between phonetic ASR errors and hallucinations lies in the severity of the latter. Phonetic ASR errors appear as badly transcribed words or phrases, especially when utterances are phonetically similar. They are being evaluated as a number of phonetic substitutions, insertions and deletions. On the contrary, the hallucinatory output does not have phonetic or semantic connection with the source utterance, even though it is often fluent and coherent. This third aspect is crucial to differentiate hallucinations from random repetitions and word salad.
We propose a framework to differentiate hallucinations from other ASR errors. Since hallucinations should resemble probable model outputs, the fluency measure should be high and semantic connection to reference should be low.
We use cosine similarity to estimate if the reference and the output are semantically related. We define hallucinations as semantically disconnected outputs. Therefore, they are the errors with the lowest cosine similarity and simultaneously high WER.
To evaluate sentence fluency, we use perplexity, which gives intuition of how viable is the sentence according to LLM. We find that results returned by Flan T5 Small contain less extreme values of perplexity values than GPT2, hence the decision to choose the former model for the fluency evaluation.
The model used for all the experiments is an encoder-decoder transformer provided by fairseq.
We present the method to induce hallucinations with random noise injection to the source utterance.
IMO, 1) phonetic mistakes may significantly change the semantics and hence give low cosine similarity, so this is not unique to hallucinaions, 2) non-phonetic mistakes (hallucinaions) are not necessarily related to the large change in semantics, and 3) fluency cannot be measured by perplexity, because gramatically correct and meaningful text is not always associated with high probability in any corpus.


# Exploring the limits of decoder-only models trained on public speech recognition corpora
We train Decoder-Only Transformer for ASR (DOTA) solely using cross-entropy loss (inspired by their success as language models) on public ASR data alone.
The audio input to the model is a sequence of audio frames followed by text token embeddings. We find that bidirectionality over audio frames is critical to high performance across model scales. For simplicity, we used sinusoidal positional embeddings.
We vary over a wide range of hyperparameters, including augmentations and the datasets included in the training set. For example, augmentations are not very important.
Our training data consists of MultilingualLibriSpeech (English), PeoplesSpeech, GigaSpeech, SPGISpeech, CommonVoice 11.0, LibriSpeech, Fisher, TedLium 3, AMI, FLEURS (English), VoxPopuli (English), LJ Speech, VoiceMail, VCTK. This resulted in 93K hours of speech-text pairs. We normalized the transcripts using Whisper’s EnglishTextNormalizer module which converts text to lower case, removes punctuation and applies several other case-by-case transformations. We further remove newline character and insert space between consecutive digits. We then tokenize the text using bert-base-uncased tokenizer.
DOTA outperforms Whisper large-v3 on 7 out of 15 test sets. As our models are trained on these sets, the errors rates are low.
IMO, the most important thing is that the test sets are in-domain for DOTA and out-of-domain for Whisper, so the comparison is by no means fair.


# WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling
Current mainstream discrete speech representations are divided into semantic and acoustic tokens (IMO, probably they mean speech-related tokens as "semantic"; however, HuBERT is their example of semantic tokens, but its design is not specific to speech, it's all about the training data). Semantic tokens often lack acoustic information, necessitating multi-stage cascades in downstream models to generate raw waveforms. Acoustic tokens can uniformly model speech, music, and audio. A robust acoustic tokenizer should at least maintain the encoder-VQ-decoder structure - this indicates that the Codec model should primarily function as a Tokenizer and De-Tokenizer. Additionally, the temporal dimension of codecs matter.
We argue that a single quantizer layer fundamentally differs from multiple quantizers. When the number of quantizers exceeds one, downstream models (that use this quantization to perform various tasks) require additional design efforts, while with a single quantizer, speech modalities can be directly autoregressively embedded into large multimodal models, such as LLama.
We introduce WavTokenizer, a discrete acoustic codec model. Our model is built on the framework of VQ-GANs. WavTokenizer passes the raw audio X through three modules. 1) a CNN+LSTM encoder network that outputs a latent feature representation Z, 2) A single quantizer discretizes Z to generate a discrete representation Z_q, 3) an improved decoder that reconstructs the audio signal ̃X from the compressed latent representation Z_q.
The model is trained end-to-end with 4 losses (eq. 3-6). The quantizer loss penalizes the distance between Z and Z_q. The mel-spectrum reconstruction loss penalizes the distance between Mel(X) and Mel(̃X). We also apply a perceptual loss in the form of discriminators operating at different resolutions, with the adversarial loss (a hinge loss over the logits of these discriminators) and the feature matching loss, penalizes the distance between feature maps in distriminators, obtained from X and from ̃X.
The goal of WavTokenizer is to compress speech representations into the codebook space of a single quantizer. Expanding the codebook space can reduce information loss caused by compressing the hierarchical RVQ structure into a single quantizer. We expanded the codebook space from 2^10 to 2^14. We adjusted the number of cluster centers to 200 to align with the larger codebook space. During training, each input’s selected code is updated using an EMA with a decay of 0.99, and codes unassigned for several batches are replaced with input vectors randomly sampled from the current batch. This forced activation strategy helps ensure effective utilization of the large codebook space.
In decoder, we achieve waveform upsampling through inverse Fourier transform. We also introduced an attention module in the decoder.


# Framework for Curating Speech Datasets and Evaluating ASR Systems: A Case Study for Polish
We collect an open benchmark of 24 openly available datasets for Polish ASR.
We perform the most extensive comparison to date of ASR systems for the Polish language. Significant variations across different systems, datasets, and speaker demographics were discovered.


# Transformer-based Model for ASR N-Best Rescoring and Rewriting
Previos work in ASR error correction  focused exclusively on either reranking the N-best hypotheses, i.e., "rescoring", or overriding the 1-best with its predicted corrections, i.e., "rewriting".
We propose Transformer Rescore Attention (TRA) model capable of both rescoring and rewriting ASR hypotheses. TRA does not require acoustic representations as input and can operate as a standalone model outside of on-device ASR, the acoustic representations never leaves the device hence preserving privacy.
We also propose a new Matching Query Similarity Distribution (MQSD) objective, that can work well with cross-entropy based training to perform both rescore and rewrite tasks.
TODO


# Careless Whisper: Speech-to-Text Hallucination Harms
We evaluate Whisper’s transcription performance on the axis of "hallucinations", defined as undesirable generated text "that is nonsensical, or unfaithful to the provided source input".
We set the sampling temperature parameter to 0.
Roughly 1% of Whisper transcriptions from AphasiaBank dataset contain entire hallucinated phrases. Whisper hallucinates entire made-up sentences when no one is speaking in the input audio files.
38% of hallucinations are harmful or concerning in some way (as opposed to innocuous and random), such as explicit portrayals of physical violence or death, or false authority like thanking viewers or specific groups, and linking to websites that misrepresent the speaker source (table 1).
Longer pauses in spoken speech (thereby, with longer periods of background noise in the audio file) could result in more hallucinations due to Whisper being seeded by noise rather than speech (fig. 3).
Notably, we found no evidence of hallucinations in competing speech recognition systems such as Google Speech-to-Text or the latest Google Chirp model. As such, we believe hallucinations to currently be an OpenAI-specific concern.
Our work compares a relatively small set of aphasia speakers to control group speakers in a setting where they are being asked a standard slate of interview questions. If a broader set of topics were to be discussed, it is possible that the scope of hallucinations would be widened.
Another one category of hallucinations is the appearance of other languages. For example, Whisper is prone to generating non-English transcriptions even when provided an argument indicating that the target language is English.
We hypothesize on two underlying mechanisms that likely result in these hallucinations.


# YODAS: Youtube-Oriented Dataset for Audio and Speech
We introduce YODAS (YouTube-Oriented Dataset for Audio and Speech), a large-scale, multilingual dataset comprising currently over 500k hours of speech data in more than 100 languages
YODAS includes: 1) the manual subset of 86,400 hours of audio data paired with manual transcriptions, 2) the automatic subset of 335,845 hours of audio data with automatic transcriptions from YouTube, 3) The unlabeled subset of 144,174 hours of raw audio data, devoid of any transcription.
English emerges as the most prevalently used language, with Spanish and Russian occupying the second and third positions, respectively, when assessed based on duration. The automatic subset has only a very limited number of languages (14 languages) compared with the manual subset (140 languages).
In the automatic subset, most utterances are short and have little variance. This is because the automatic subtitle frequently divides long utterances into small chunks to help viewers to follow subtitles easier.
We focus on monolingual speech recognition, and build simple baseline models for the top-25 languages in the manual subset. Our baseline is a linear layer randomly initialized on top of the pre-trained XLSR representations, which is then optimized with the CTC loss. The subword vocabulary is prepared with BPE using SentencePiece, where we use 300 as the vocabulary size for most languages except 5000 for Mandarin and 3000 for Japanese. For simplicity, we do not perform speech augmentation. The decoding is done greedily without any language models. Table 5 displays our results for monolingual speech recognition. We observe that languages possessing a larger BPE vocabulary size tend to correspond with higher CER (is it a random split validation?). Models trained on the manual subset yield significantly superior performance compared to those trained on the automatic subset.


# FLEURS-R: A Restored Multilingual Speech Corpus for Generation Tasks
We introduce FLEURS-R speech-text dataset. It maintains an N-way parallel speech corpus in 102 languages as FLEURS, with improved audio quality and fidelity by applying the speech restoration model Miipher.
The goal of the dataset is to catalyze speech generation research in low-resource languages, since in the original FLEURS dataset all recordings are kept as they-are, either from quiet or noisy environment, while speech generation models are requested to produce high quality speech.
Since Miipher supports only English, we replaced the acoustic feature extractor from w2v-BERT to the 2-billion parameter non-fine-tuned Universal Speech Model (USM). It is known that deeper layers tend to lose detailed and local acoustic information; therefore, we used the intermediate feature from the 13th of 32th layers based on preliminary experiments. Also, neither text nor speaker conditioning improved the reconstruction accuracy. Consequently, both speaker encoder and PnG-BERT text encoder were removed from the new Miipher network architecture.
To identify successfully restored samples, we performed ASR-based filtering.


# SALSA: Speedy ASR-LLM Synchronous Aggregation
The current methods to enhance ASR models with LMs suffer either from the exiensive training requirements, or from high decoding latencies due to second-pass rescoring in ASR error correction. Also, many current methods rely on the n-best predictions and will not fare well on low-resource languages owing to large errors in the n-best predictions.
We propose SALSA, a lightweight method to integrate LM model with ASR model (we use Whisper and LLama-2). It keeps both backbones frozen and only train projection layers. It can be used to integrate any pretrained decoder-only LM with a pretrained encoder-decoder ASR model using small amounts of labeled speech in the target languages.
We are also the first to apply utilize LMs for ASR of a diverse set of low-resource languages (not only English ASR).
We select N different ASR decoder layers and N different LM decoder layers and connect them one-to-one: for each pair, a trainable mapping R^m -> R^m processes an ASR hidden state, and the result is added to the LM hidden state (fig. 1). The LM keeps generating tokens until a valid text piece recognizable by ASR’s tokenizer is formed (often for low resource languages the tokenizers for LM and ASR can use different multi-token sequences to encode a single character in the target language). Then the just generated text is re-tokenized with the ASR’s tokenizer. Thus, in SALSA both decoders (ASR and LM) move forward in tandem albeit having different tokenizations.
So, the authors generate with LM until a valid utf-8 character if formed, and then re-tokenize the resulting text with ASR tokenizers.


# Orthogonality and isotropy of speaker and phonetic information in self-supervised speech representations
We study how information is represented in self-supervised speech representations, beyond just assessing the linear separability of classes. We use a geometric approach, that is widely used for analyzing self-supervised models of text.
We develop a new measure for analyzing high-dimensional distributions, the Cumulative Residual Variance (CRV). Given datasets X and Y embedded in the same embedding space, the CRV of X w.r.t. Y (denoted as X\Y) measures the degree to which the principal components of Y are orthogonal to those of X. Meanwhile, X\X is a measure of the isotropy of X — the degree to which X effectively utilizes all dimensions of the embedding space, i.e., has uniform covariance.
Comparing to the previous work "Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces", the CRV measure allows us to better quantify orthogonality.
On English LibriSpeech we show that, unlike randomly initialized models, all trained models have a high degree of orthogonality between the speaker and phonetic subspaces. For all 6 trained models, the accuracy of a phone classifier trained on the model representations is significantly correlated with the CRV between the two subspaces.
It has been argued in the NLP literature that higher isotropy is desirable in an embedding space. However, we did not find strong evidence for this hypothesis. Instead, having evenly distributed centroids is more important for classification accuracy in these models than having evenly distributed frame representations.


# Speech Robust Bench: A Robustness Benchmark For Speech Recognition
We propose Speech Robust Bench (SRB), a benchmark for evaluating the robustness of ASR models to 69 input perturbations that ASR models may encounter in the physical and digital world (fig. 2). The perturbations are of two broad types: 1) non-adversarial and 2) adversarial.
SRB is agnostic to the evaluation data and can be used with any dataset that contains utterances and reference transcripts, however, we recommend using datasets with high-quality clean audio and accurate transcripts so that pre-existing corruptions in the dataset do not confound the robustness metrics. Our evaluation in uses clean speech from Librispeech.
We measure two aspects of robustness: the transcription accuracy (WER) and the stability under randomized perturbations. When aggregating WER over multiple perturbations, we normalize the WER of the target model by the WER of a baseline model (following the methodology of "Benchmarking neural network robustness to common corruptions and perturbations" on images domain). Doing so penalizes errors on “easy” corruptions more than errors on “harder” corruptions. This normalized metric is called Normalized WER (NWER). Also SRB measures the prediction stability of the model by computing the variance in the WER caused by corrupting the signal with multiple corruption samples drawn from the same distribution. We call this metric WER Variance (WERV).
On English speech, Whisper is able to withstand more severe corruptions better than other models. However, it is outperformed by other, smaller, models on several perturbations. It is rather surprising that despite being trained on more than ten times the amount of data, wsp-lg is outperformed by both hubt-lg and w2v2-lg-slf on fairly common perturbations such as RIR, resampling, and tempo reduction. Some models (hubt-lg, w2v2-lg-slf, ds) are more stable on Gaussian noise, while others ( wsp-lg, w2v2-bs, w2v2-lgrob) are more stable on environmental noise. Larger models tend to be more robust than smaller models, even if the latter are trained on significantly more data.
On non-English (Spanish) speech, it is interesting that despite having more parameters and being trained on 10× more data, wsp-lg is outperformed by w2v2-lg-es, thus indicating that simply scaling the model and training data is not sufficient to achieve robustness, particularly in the multi-lingual setting. So, wsp-lg is not the most robust model on Spanish, and struggles on simple perturbations, particularly on RIR (room impulse response).
We observe noticeable disparities in the robustness across various demographic subgroups, for example, RIR and adversarial attacks disproportionately degrade the performance of models for female speakers.


# ProGRes: Prompted Generative Rescoring on ASR n-Best
ASRs are not trained on enough data to deeply capture linguistic information, resulting in challenges when transcribing unknown words and named entities. Language models are commonly used to enhance ASR performance by ensuring that transcriptions maintain linguistic plausibility. Rescoring can be performed in two ways: either during the decoding process, where partial hypotheses are rescored, or after decoding, where the n-best alternatives are rescored. Traditionally, rescoring involved using simple language models, such as n-grams and word graphs. However, rescoring provides minimal benefits, even with LLMs, when the correct transcription is not among the top n hypotheses.
We propose PROmpted Generative REScoring (ProGRes). First, for each audio signal, we extract the n-best hypotheses from a pretrained ASR model. We then prompt an LLM to generate a more diverse set of hypotheses. The LLM-generated hypothesis are added to form an extended set of hypotheses. Then we calculate LLM score and ASR score (either log-likelihood or CTC likelihood) for each hypothesis and combine both scores to select the best hypothesis.


# A Large-Scale Evaluation of Speech Foundation Models
We extend our SUPERB benchmark (see "SUPERB: Speech processing Universal PERformance Benchmark") with the following contributions:
1) We provide a complete platform featuring an online leaderboard supporting submissions
2) We scale the evaluation from the original 14 models [34] to 33 models
3) We observe that the learnable weighted-sum over the frozen layers of the SSL model is better than the conventional evaluation protocol: using the frozen last layer. Furthermore, individual single-layer benchmarking can sometimes yield even better results.
4) We confirm that the layer weights learned by the weighted-sum protocol do not reflect the layer performance precisely across SUPERB tasks (as in "Layer-Wise Analysis of a Self-Supervised Speech Representation Model").
5) We suggest to conduct statistical test when comparing to our baseline numbers.


# Towards Robust Speech Representation Learning for Thousands of Languages
Currently, the best performing models like Whisper, Google USM, w2v-BERT 2.0 v1 and and w2vBERT 2.0 v2 are all trained on fully closed data. Whisper and w2v-BERT 2.0 v1/v2 only report pre-training data quantity and the languages covered. The USM report includes much more information about their data sources, but the model checkpoints remain unreleased. XLSR 53 and XLS-R 128 came with checkpoints and only use publicly accessible datasets but did not release training code. MMS released checkpoints but did not release their training code and crawled data. WavLabLM and MR-HuBERT released code and checkpoints but operated on a smaller scale.
For our work, will publicly release all of our heavily optimized training code, along with the training configurations and checkpoints for XEUS. We use publicly accessible datasets and release all of the additional pre-training data that we crawled. We release 200+ intermediate checkpoints and training logs for further research in the training dynamics.
Robustness to noisy data is relatively unexplored in SSL research. This is important for multilingual models, since the available recordings of low-resource languages tend to be particularly noisy.
We propose XEUS (pronounced Zeus) — a Crosslingual Encoder for Universal Speech. Comparing to Meta’s MMS from "Scaling speech technology to 1,000+ languages", we scale the language coverage, use more powerful model architectures and training objectives.
We curate the data from 37 existing corpora (150+ languages, 1.074 million hours of data in total, see table 2 and table 11 - main sorces are YODAS and VoxPopuli, both around 400K hours). We thus ensure diversity, including but not limited to spontaneous speech, accented speech, code-switching, indigenous languages, and singing voices.
To increase language coverage, we add 3 more data sources:
1) We reproduce the MMS-unlab dataset, which was not publicly released (see "Scaling speech technology to 1,000+ languages"). Like the original, we crawl religious audiobooks from the Global Recordings Network. Since we use it for SSL instead of language identification, we do not filter out languages with low amounts of data. We also perform VAD with an energy-based detector instead of a neural model, which is more computationally expensive and likely less robust to unseen languages. This leads to a total of 6,700 hours of data across 4,023 languages.
2) We crawl data from WikiTongues, where each 2-20 minute recording contains 1-2 speakers casually speaking a particular language/dialect.
3) We collect a Jesus Dramas corpus: the "Story of Jesus" multi-speaker audio drama on many languages, totalling 645 hours.
XEUS is an E-Branchformer encoder consisting of a convolutional feature extractor and 19 E-Branchformer layers. Convolution augmented models achieve superior SSL performance, and we choose the E-Branchformer over the Conformer due to the former’s relative ease of training and superior downstream performance.
Training combines HuBERT’s masked prediction (with cross entropy loss), WavLM’s denoising objective, and a new dereverberation objective (fig. 2). We also conduct ablations at a smaller scale.
To obtain the target phonetic pseudo-labels for HuBERT masked prediction, we first extract encoded representations from a pre-trained WavLabLM MS model. The representations are then clustered using k-means, with k = 2048. The data used for the feature extraction and clustering is a subset of our training data.
We also integrate the acoustic denoising task proposed by WavLM. During training, an input utterance has a probability 0.2 to be augmented with either random noise from the Deep Noise Suppression Challenge, or another utterance in the batch as interference. Target labels are obtained solely from uncorrupted speech.
We also propose a novel SSL objective: with probability 0.3 we simulate reverberant conditions in the input audio while the target pseudo-labels are again left untouched. It is possible to apply both the noise and reverberation. We use a Room Impulse Response (RIR), see sec. 4.2.
XEUS is pre-trained on 64 A100 GPUs using the ESPnet toolkit. We perform a two passes through the training set.
We compare XEUS with 3 SOTA multilingual SSL models: XLS-R 128, MMS, and w2v-BERT 2.0 v2. XEUS is the overall best performing model on ML-SUPERB and is competitive on FLEURS (tables 3, 4).
We benchmark XEUS on the English-only SUPERB, comparing to WavLM, the SOTA model on the SUPERB leaderboard for almost all tasks. XEUS consistently reaches if not surpasses SOTA scores across a variety of tasks, obtaining the highest score in 4 English-only tasks (Keyword Spotting, Speaker Diarization, Emotion Recognition, Speech Recognition), despite its curse of multilinguality.
Also, resynthesized speech from XEUS is higher quality than that from both WavLM and w2v-BERT 2.0 v2 across all metrics (with unit-to-speech HiFiGAN vocoders trained for speech synthesis).


# Lhotse: a speech data representation library for the modern deep learning ecosystem
Speech data is notoriously difficult to work. Recordings come in many flavors, audio is encoded with a variety of codecs, the meta-data comes with a different schema for each dataset. Kaldi introduced a standard representation for all datasets, however its learning curve is steep.
We present Lhotse, a speech data representation library. It is one of the three libraries that constitute the next-generation Kaldi framework (the remaining two are k2 for differentiable, GPU-accelerated weighted finite state automata (WFSA) algorithms; and Icefall that contains simple, reproducible recipes for training and evaluating speech models).
Lhotse provides standard data preparation recipes for publicly available corpora (over 30 at the time of writing).
Lhotse defines several types of speeech manifests:
1) `Recording` abstracts away from the physical location of the audio data. It allows reading audios stored on a local disk, remote URL or a cloud storage service, possibly encoded in different format, with `Recording.load_audio()`.
2) `SupervisionSegment` denotes a time span in the Recording that has some associated meta-data usable for supervised training (start, duration, speaker ID, gender, age, transcript, etc).
3) `Features` helps to work with pre-computed features for model training or inference. It contains tensor shape, frame shift, `Recording` ID etc. Similarly to `Recording`, it abstracts away from the storage mechanism and allows to read/write to a local file-system, a cloud storage, and more. Lhotse supports both on-the-fly and pre-computed feature extraction and leverages lilcom, a feature compression engine which lossily compresses floating-point NumPy arrays into byte strings.
4) `Cut` manifests are the core contribution of Lhotse. They may be viewed as "windows" with a certain offset and duration in a `Recording` that have zero or more `SupervisionSegments`. Most operations performed on cuts are lazy - whether it’s mixing, truncation, padding, or augmentation. `Cut` provides greater data preparation flexibility than was possible with Kaldi - it allows to construct training examples with additional acoustic context for each utterance (e.g., background noises in a telephone conversation that could help the network adapt) or even additional speech context (e.g., modelling contextual dependencies between utterances in a podcast).
5) `CutSet` is a collection of cuts. All speech corpora are represented as CutSets. It has over 30 methods that simplify padding, sub-setting, truncating, extracting features, or visualizing and listening to in Jupyter notebooks. `CutSet` also supports data augmentation that correctly adjusts the relevant meta-data, such as speech segments duration. `CutSet` has a few methods for converting long recordings into shorter cuts (fig. 1).
Note that Lhotse is not a feature extraction or augmentation library (it leverages external implementations for these tasks), and is not a framework for training, it just provides a set of tools.
Lhotse implementы the PyTorch data API: `Dataset`, `Sampler`, and `DataLoader`. Very large datasets (tens of thousands of hours) are seamlessly handled with minimal memory usage. In our implementation `Sampler` not only provides indices, it "owns" the CutSets with training data, determines the batch size dynamically, based on constraints such as the maximum total speech duration in a mini-batch, and `Dataset` simply acts as a function that transforms a mini-batch `CutSet` into a mini-batch tensor. Lhotse implements several CutSamplers. `SingleCutSampler` is used where the model works with single utterances, `CutPairsSampler` uses two CutSets with matching cut IDs (for example, for voice conversion or speech translation), `BucketingSampler`augments sampler types by stratifying the data into similar-duration buckets to minimize the padding. `ZipSampler` draws batches from multiple samplers and combines them together, which is useful in diversifying the data from multiple sources (domains) during the training.


# Audio Set: An ontology and human-labeled dataset for audio events
We present Audio Set, a dataset and ontology of audio events that endeavors to provides comprehensive coverage of real-world sounds at ImageNet-like scale.
Our final list contains 632 audio event categories, arranged in a hierarchy with a maximum depth of 6 levels; an example from among the eight level-6 nodes is "Sounds of things" → "Vehicle" → "Motor vehicle" → "Emergency vehicle" → "Siren" → "Ambulance (siren)".
For each class we provide a description, typically one or two sentences, and examples. Of the 632 categories, 56 are "blacklisted" because they have turned out to be obscure or confusing (e.g., "Sounds of things").
Another 22 nodes are marked "Abstract" (e.g., "Onomatopoeia"), meaning that they exist purely as intermediate nodes to help structure the ontology, and are not expected to ever be used directly as labels.
The Audio Set YouTube Corpus consists of labeled YouTube segments with one or more ontology class labels. It includes 1,789,621 segments (4,971 hours). Single segments can have multiple labels (on average 2.7 labels per segment).
Human raters were presented with both the video and audio components (audio-only presentation was found to be more difficult) and asked to independently rate the presence of one or more labels. Each segment was rated by three raters and a majority vote was required to record an overall rating. For speed, a segment’s third rating was not collected if the first two ratings agreed for all labels. The raters were unanimous in 76.2% of votes. The "unsure" answer was rare.
We provide maximally-balanced train and test subsets (from disjoint videos), chosen to provide at least 50 positive examples (in both subsets) for as many classes as possible.
There were some categories for which we were unable to find enough positive examples to fully populate the dataset, but this proportion is now very small (and shrinking).
We have trained a simple baseline system.


# AST: Audio Spectrogram Transformer
Motivated by the success of purely attention-based models in the vision domain, it is reasonable to ask whether a CNN is still essential for audio classification.
We introduce the Audio Spectrogram Transformer (AST), a convolution-free, purely attention-based model (fig. 1). It is applied to an audio spectrogram, which is split into a sequence of 16×16 patches with overlap, and then linearly projected to a sequence of 1-D patch embeddings with learnable positional embeddings. An additional CLS token is prepended to the sequence. The resulting sequence is then input to the Transformer encoder. The output of the CLS token is used for classification with a linear layer with sigmoid activation maps. (IMO, sigmoid is probably used because AudioSet is a multi-label classification dataset; but what about other tasks?)
Strictly speaking, the patch embedding layer can be viewed as a single convolution layer with a large kernel and stride size, and the projection layer in each Transformer block is equivalent to 1×1 convolution. However, the design is different from conventional CNNs, and these Transformer models are usually referred to as convolution-free.
Transformer needs more data to train than CNNs. We are able to transfer the 2D spatial knowledge from a pretrained ViT (slightly modified for compatibility) to the AST. mageNet pretrained AST noticeably outperforms randomly initialized AST, especially when the training data volume is smaller. We initialize AST with DeiT weights, and further perform ImageNet distillation.
Using 128×2 rectangle patches leads to better performance than using 16×16 square patches when both models are trained from scratch. However, there is no 128×2 patch based ImageNet pretrained models, so using 16×16 patches is the optimal solution. Also smaller size patches lead to better performance.
AST naturally supports variable-length inputs and can be applied to different tasks.


# Global optimization of a neural network-hidden Markov model hybrid
We propose NN-HMM hybrid model for ASR. The outputs of NN constitute the observation sequence for the continuous density HMM (Hidden Markov Model). The parameters of both NN and HMM are optimized, and NN is optimized by backprop through HMM.


# Deep Belief Networks for phone recognition
A SOTA ASR system typically uses Hidden Markov Models (HMMs) to model the sequential structure of speech signals, with local spectral variability modeled using mixtures of Gaussian densities. The first assumption is that the hidden state sequence can be well-approximated using a first order Markov chain where each state S_t at time t depends only on S_{t−1}. Second, observations at different time steps are assumed to be conditionally independent given a state sequence. Although these assumptions are not realistic, they enable tractable decoding and learning even with large amounts of speech data.
We apply Deep Belief Networks (DBNs) to model the spectral variabilities in speech. DBNs are probabilistic generative models that are composed of multiple layers of stochastic latent variables with Restricted Boltzmann Machines (RBMs, a particular type of Markov Random Field) as their building blocks. DBNs have a greedy layer-wise unsupervised learning algorithm as well as a discriminative finetuning procedure for optimizing performance on classification tasks.
In order to apply DBNs with fixed input and output dimensionality to phone recognition, a context window of n successive frames of feature vectors is used to set the states of the visible units of the lower layer of the DBN.
Phone recognition experiments were performed on the TIMIT corpus.


# Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups
This paper provides an overview of ASR progress and represents the shared views of four research groups.
Most current speech recognition systems use hidden Markov models (HMMs) to deal with the temporal variability of speech and Gaussian mixture models (GMMs) to determine how well each state of each HMM fits a frame or a short window of frames of coefficients that represents the acoustic input. With enough components, GMMs can model probability distributions to any required level of accuracy and they are fairly easy to fit to data using the EM algorithm.
The acoustic input is typically represented by concatenating Mel Frequency Cepstral Coefficients (MFCCs) or Perceptual Linear Predictive coefficients (PLPs).
The recognition accuracy of a GMM-HMM system can be further improved if it is discriminatively fine-tuned after it has been generatively trained to maximize its probability of generating the observed data.
GMMs have a serious shortcoming – they are statistically inefficient for modeling data that lie on or near a non-linear manifold in the data space. For example, modeling the set of points that lie very close to the surface of a sphere only requires a few parameters using an appropriate model class, but it requires a very large number of diagonal Gaussians or a fairly large number of full-covariance Gaussians. We believe that other types of model may work better than GMMs for acoustic modeling if they can more effectively exploit information embedded in a large window of frames.
DNNs have the potential to learn much better models of data that lie on or near a non-linear manifold. An alternative way is to use a feedforward NN that takes several frames of coefficients as input and produces posterior probabilities over HMM states as output.
Two decades ago, researchers achieved some success using NNs with a single layer of non-linear hidden units to predict HMM states from windows of acoustic coefficients. At that time, however, neither the hardware nor the learning algorithms were adequate for training NNs with many hidden layers on large amounts of data and the performance benefits of using NNs with a single hidden layer were not sufficiently large to seriously challenge GMMs. As a result, the main practical contribution of NNs at that time was to provide extra features in tandem or bottleneck systems.
The paper starts by describing the two-stage training procedure that is used for fitting the DBN (fig. 1). In the first stage (generative pre-training), layers of feature detectors are initialized, one layer at a time, by fitting a stack of generative models, each of which has one layer of latent variables. These generative models are trained without using any information about the HMM states that the acoustic model will need to discriminate. In the second stage, each generative model in the stack is used to initialize one layer of hidden units in a DNN and the whole network is then discriminatively fine-tuned to predict the target HMM states. These targets are obtained by using a baseline GMM-HMM system to produce a forced alignment.
We review exploratory experiments on the TIMIT database. The DNNs worked well on all of these tasks when compared with highly-tuned GMM-HMM systems and on some of the tasks they outperformed the state-of-the-art by a large margin.


# End-to-end Continuous Speech Recognition using Attention-based Recurrent NN: First Results
The authors propose seq2seq RNN with attention for ASR. The RNN decoder directly emits a stream of phonemes. This model is closely related to the RNN Transducer, however, with an attention mechanism.
The accuracy on TIMIT phoneme recognition is comparable to the SOTA DNN-HMM systems and is slightly worse than the best reported error rates obtained using RNNs.
However, the model is easy to implement, tune and apply. It requires a narrow beam search, and its accuracy deteriorates very slightly when greedy search.
Fig. 2 shows alignments produced by the model: (a) when the alignment was successfully encouraged to be monotonic, and (b) when the model is free to select any frame in the input sequence. In (b), we observe how the absence of the learned preference for monotonicity makes the model confused by the repeated occurrence of the phonemes “cl k”, and to a lesser degree, by the repetition of “w”. (IMO, interestingly, Whisper is also subject to this).
IMO, a good Background section with a review of previous approaches. However, there is nothing about the justification of predicting phonemes instead of graphemes.


# EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding
ASR has traditionally leveraged the HMM/GMM paradigm for acoustic modeling. HMMs act to normalize the temporal variability, whereas GMMs compute the emission probabilities of HMM states. In recent years, the performance of ASR has been improved dramatically by the introduction of deep neural networks (DNNs) as acoustic models. In the hybrid HMM/DNN approach, DNNs are used to classify speech frames into clustered context-dependent (CD) states (i.e., senones). However, at first, acoustic modeling typically requires various resources such as dictionaries and phonetic questions. Second, in the hybrid approach, training of DNNs still relies on GMM models to obtain initial frame-level labels, while building GMM models normally goes through multiple stages. Third, the development of ASR systems highly relies on ASR experts to determine the optimal configurations of a multitude of hyper-parameters, for instance, the number of senones and Gaussians in the GMM models.
We need to reduce the complexity of ASR. We focus on end-to-end ASR, i.e., modeling the mapping between speech and labels (words, phonemes, etc.) directly without any intermediate components (e.g., GMMs). Research on end-to-end ASR faces two major obstacles. First, it is challenging to incorporate lexicons and language models into decoding. When decoding CTC-trained models, past work has successfully constrained search paths with lexicons. However, how to integrate word-level language models efficiently still is an unanswered question. Second, the community lacks a shared experimental platform for the purpose of benchmarking.
We present our Eesen framework which drastically simplifies the existing pipeline to build SOTA ASR systems. In Eesen, RNN predicts phonemes or characters with CTC objective.
A distinctive feature of Eesen is a generalized decoding method based on weighted finite-state transducers (WFSTs). In this method, individual components (CTC labels, lexicons and language models) are encoded into WFSTs, and then composed into a comprehensive search graph.
Eesen results in superior performance than the existing end-to-end ASR pipelines, being on a par with strong hybrid HMM/DNN baselines.


# Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion
We use bidirectional encoder-decoder LSTM (fig. 3) to improve SOTA on Grapheme-to-Phoneme Conversion.
We experiment on the CMUDict, NetTalk, and Pronlex datasets.


# Phonetisaurus: Exploring grapheme-to-phoneme conversion with joint n-gram models in the WFST framework
Grapheme-to-Phoneme (G2P) conversion is an important problem in both the areas of ASR and TTS. In the case of ASR, the true vocabulary is often dynamic in nature. This means that new words, or new pronunciation candidates for existing words may need to be added to the system on a regular basis. Analogous problems arise in the case of TTS (IMO not a clear explanation why do we need G2P).
We introduce Phonetisaurus, an open-source G2P conversion toolkit. We syntesize the most effective components of previously proposed solutions in the literature, with a clear focus on achieving a balance between speed, accuracy and flexibility.


# Very Deep Convolutional Networks for End-to-End Speech Recognition
The seq2seq model with attention sidesteps the complicated machinery developed for classical ASR, because it is not restricted by the classical independence assumptions of HMM and CTC models.
While very deep CNNs have been successfully applied to ASR, recently there have been several advancements in the CV community on very deep CNNs hat have not been explored in the speech community.
In our deep CNN speech model based on Listen, Attend and Spell (LAS) we use 1x1 convolutions (Network-in-Network), BN, ResNets, and Convolutional LSTM that use convolutions to replace the inner products within the LSTM unit. The model learns to transcribe an audio sequence to a word sequence, one character at a time.
We experiment with the WSJ ASR task and achieve 10.5% WER without any dictionary or language.


# Towards better decoding and language model integration in sequence to sequence models
Seq2seq ASR  networks can typically be decomposed into an encoding module that transforms its inputs into a hidden representation, a decoding (spelling) module which emits target sequences and an attention module that computes a soft alignment between the hidden representation and the targets. This discriminative training mode is fundamentally different from the generative "noisy channel" formulation used to build classical SOTA ASR systems. Discriminative training allows seq2seq models to focus on the most informative features. However, it also increases the risk of overfitting to those few distinguishing characteristics.
We analyse an attention-based seq2seq ASR system that directly transcribes recordings into characters.
We observe two shortcomings: overconfidence in predictions (that reduces the diversity of transcripts obtained using beam search) and a tendency to produce incomplete transcriptions when external LMs are used (model skips some words).
Model overconfidence is promoted by the the cross-entropy training criterion. This leads to very peaked probability distributions, effectively preventing the model from indicating sensible alternatives to a given character, such as its homophones. Model overconfidence can have two consequences. First, next-step character predictions may have low accuracy due to overfitting. Second, overconfidence may impact the ability of beam search to find good solutions and to recover from errors.
We first investigate the impact of confidence on beam search by varying the temperature of the SoftMax function. As temperature increases beam search finds better solutions, however care must be taken to prevent truncated transcripts: we constrained the search to emit the EOS token only when its probability was within a narrow range from the most probable token.
A solution to model overconfidence is label smoothing. Originally label smoothing is uniform. Better results can be obtained with unigram smoothing which distributes the remaining probability mass proportionally to the marginal probability of classes (see "Regularizing neural networks by penalizing confident output distributions"). We propose a neighborhood smoothing scheme that uses the temporal structure of the transcripts: the remaining probability mass is assigned to tokens neighboring in the transcript. (IMO not clear).
When training with neighborhood smoothing, greedy decoding leads to nearly 3 percentage smaller error rate. Second, the entropy of network predictions is higher, allowing beam search to discover good solutions without the need for temperature control. Moreover, since model is trained and evaluated with the same temperature 1 we didn’t have to control the emission of EOS token.
When a language model is used wide beam searches often yield incomplete transcripts. With narrow beams, the problem is less visible due to implicit hypothesis pruning.
We compare three strategies designed to prevent incomplete transcripts. The first strategy doesn’t change the beam search criterion, but forbids emitting the EOS token unless its probability is within a set range of that of the most probable token. Hoever, this is inefficient against omissions in the middle of the transcript. Alternatively, beam search criterion can be extended to promote long transcripts, but in this case beam search is looping over parts of the recording and additional constraints are needed. To prevent looping we propose to use a coverage term that counts the number of frames that have received a cumulative attention greater than some value. This prevents looping because once the cumulative attention bypasses the threshold τ a frame is counted as selected and subsequent selections of this frame do not reduce the decoding cost.
We observe that at large beam widths constraining EOS emissions is not sufficient. In contrast, both promoting coverage and transcript length yield improvements with increasing beams. However, simply maximizing transcript length yields more word insertion errors and achieves an overall worse WER.


# Grapheme-to-Phoneme Models for (Almost) Any Language
Grapheme-to-phoneme (G2P) models are typically language-specific. They are trained on a pronunciation dictionary consisting of word-pronunciation pairs. Building such a dictionary for a new language is both time-consuming and expensive, because it requires expertise in both the language and a notation system like the International Phonetic Alphabet.
Using data scraped from Wiktionary, we clean and normalize pronunciation dictionaries for 531 languages.
We develop a language-independent distance metric between IPA (International Phonetic Alphabet) phonemes.
We create two sets of g2p models for "high resource" languages and adapt them to low-resource languages through output mapping and training data mapping.


# Multitask Learning with Low-Level Auxiliary Tasks for Encoder-Decoder Based Speech Recognition
Traditional ASR systems include components like frame classifiers, phonetic acoustic models, lexicons (which may or may not be learned from data), and LMs. Recently, completely integrated end-to-end training approaches, where all parameters are learned jointly using a loss at the final output level, have become viable and popular. Typical end-to-end models are based on RNN encoder-decoders or CTC-based models. However, end-to-end training is less interpretable and ignores potentially useful domain-specific information about intermediate representations, as well as existing intermediate levels of supervision.
We use a multitask learning approach that combines the final task loss (log loss on the output labels) with losses corresponding to lower-level tasks applied on lower layers. We demonstrate this approach on an attention-based encoder-decoder LSTM character-level ASR model (fig. 1).
We use phoneme-level supervision obtained from the word-level transcriptions and pronunciation dictionary. Also, we apply sub-phonetic type of supervision at the frame level, as shown in Figure 1, using state alignments obtained from a standard HMM-based system.
Results on Switchboard and CallHome show consistent improvements over baseline attention-based models. We obtain the best performance with a combination of 2 tasks: a phoneme decoder and frame-level state loss. Analysis of model training and performance suggests that the addition of auxiliary tasks can help in either optimization or generalization.


# Massively Multilingual Neural Grapheme-to-Phoneme Conversion
Accurate grapheme-to-phoneme conversion (g2p) is important for any application that depends on the sometimes inconsistent relationship between spoken and written language. Most prominently,this includes text-to-speech and automatic speech recognition (IMO not a clear explanation why do we need G2P for ASR).
We present a neural seq-to-seq approach to g2p which is trained on spelling–pronunciation pairs in hundreds of languages. The system shares a single encoder and decoder across all languages, allowing it to utilize the intrinsic similarities between different writing systems.
For our model, the source sequences are words in the standard orthography in any language, and the target sequences are the corresponding representation in the International Phonetic Alphabet (IPA).


# Multitask Sequence-to-Sequence Models for Grapheme-to-Phoneme Conversion
A crucial component of most ASR systems is the phoneme lexicon, mapping words to their phonetic representation (e.g. Thursday → TH ER Z D EY). Training and using a G2P model is often directly integrated into the ASR training procedure, as phonetic out-of-vacabulary (OOV) words in the training set hamper the alignment of training data to its transcriptions.
we investigate how multitask learning can improve the performance of Seq2Seq G2P models. A single Seq2Seq model is trained on multiple phoneme lexicon datasets containing multiple languages and phonetic alphabets.
Multi-language learning does not show improved error rates.
Combining standard datasets and crawled data with different phonetic alphabets of the same language shows promising error reductions on English and German Seq2Seq G2P conversion.
Combining Seq2seq G2P models with standard n-grams based models yields significant improvements.


# No Need for a Lexicon? Evaluating the Value of the Pronunciation Lexica in End-to-End Models
Traditional automatic speech recognition (ASR) systems are comprised of an acoustic model (AM), a language model (LM) and a pronunciation model (PM), all of which are independently trained on different datasets. AMs take acoustic features and predict a set of sub-word units, typically context-dependent or context-independent phonemes. Next, a hand-designed lexicon (i.e., PM) maps a sequence of phonemes produced by the acoustic model to words. Finally, the LM assigns probabilities to word sequences.
How do end-to-end models perform if we incorporate a separate PM and LM into the system? This question can be answered by training an end-to-end model to predict phonemes instead of graphemes. The output of the end-to-end model must then be combined with a separate PM and LM to decode the best hypotheses from the model.
The present work is the first to explore end-to-end systems trained with phonemes for a large vocabulary continuous speech recognition (LVCSR) task, where models are directly decoded in the first-pass.
When predicting phonemes, we train our model to predict a set of 44 CI phonemes, as well as an extra <eow> token, specifying the end of a word, analogous to the <space> token in graphemes. Because it is hard to predict <eow>, we found it is better to make it optional.
Because of the homophone issue with phonemes (e.g., phoneme ey can map to the words ‘I’ or ‘eye’), using a language model, G, is critically important. There are two ways we can incorporate it during decoding. We can either process final phoneme sequences, ir incorporate LM ducing each step of the beam search (eq. 3) with additional "coverage" term to promote longer transcripts. As noted in "Towards better decoding and language model integration in sequence to sequence models", the latter way can become quite challenging if the ASR model becomes over-confident, in which case, the weight from the LM component will be ignored. In experiments we found that the first way (processing final phoneme sequences with LM) is slighly better.
Our experiments show that the performance of grapheme systems is slightly better than phoneme systems. On a multi-dialect English task we once again confirm the superiority of graphemes (see examples in tables 3, 4, 5).


# Epitran: Precision G2P for Many Languages
Epitran is a massively multilingual, multiple back-end system for G2P (grapheme-to-phoneme) transduction that takes word tokens in the orthography of a language and outputs a phonemic representation in either IPA or X-SAMPA. Out of the box, it supports 61 languages.


# A Comparison of Modeling Units in Sequence-to-Sequence Speech Recognition with the Transformer on Mandarin Chinese
Conventional ASR systems consist of three independent components: an acoustic model (AM), a pronunciation model (PM) and a language model (LM), all of which are trained independently. CD-states and CD-phonemes are dominant as their modeling units in such system. However, it has been challenged by seq2seq attention-based models, which integrate an acoustic, pronunciation and language model into a single NN.
On English ASR tasks, previous attempts have already shown that the modeling unit of graphemes can outperform that of phonemes by seq2seq attention-based model (see "No Need for a Lexicon? Evaluating the Value of the Pronunciation Lexica in End-to-End Models"), so a hand-designed lexicon might be removed from ASR systems (as we known, it is very laborious and time-consuming to generate a pronunciation lexicon). Furthermore, the latest work use the word piece models, which are sub-word units ranging from graphemes all the way up to entire words.
We investigate five modeling units for Speech-Transformer model on Mandarin Chinese ASR tasks, including CI-phonemes, syllables (pinyins with tones), words, sub-words and characters.
We confirm that the lexicon free modeling units, i.e. words, sub-words and characters, can outperform lexicon related modeling units, i.e. CI-phonemes and syllables. Character based model achieves the best result and establishes a new SOTA CER on HKUST dataset.


# Simple and Effective Zero-shot Cross-lingual Phoneme Recognition
We are fine-tuning wav2vec 2.0 to transcribe languages unseen during fine-tuning. We start from self-supervised representations trained on data in many languages (wav2vec 2.0). Next we simultaneously fine-tune the model to perform phoneme recognition on data in multiple training languages, building a global phoneme recognizer by simply considering all possible phonemes of the training languages. At inference time, we test the fine-tuned model on unseen languages using a mapping of the phonemes from the training vocabulary to the ones in the target languages. We decode with a LM to generate the final phoneme sequence.
Our approach performs on par to the recently introduced unsupervised speech recognition work ("Unsupervised speech recognition") which does not use labeled data from related languages and requires training separate models for each target language.


# Transformer based Grapheme-to-Phoneme Conversion
We are first to apply encoder-decoder transformer for Grapheme-to-Phoneme Conversion (G2P). For evaluation, the CMU pronunciation and NetTalk datasets were used (see examples in table 4)


# Using Phoneme Representations to Build Predictive Models Robust to ASR Errors
In Amazon Alexa, Apple Siri or Google Home, the Spoken Language Understanding (SLU) is usually performed in two steps: first an Automatic Speech Recognition (ASR) is used to transcribe human speech; then Natural Language Understanding (NLU) models are applied on ASR transcriptions to interpret users’ requests. Applying NLU it on ASR transcriptions poses new challenges, as ASR systems often generate transcriptions with errors that can cause failures in downstream applications of virtual assistants, such as intention classification or slot filling.
ASR errors are just an outcome of a phonetic confusion, causing a phrase in a human speech to be incorrectly transcribed to a "quasi-oronym", i.e., phrases with different meanings that sound very similar. Therefore, classic approaches that operate on word or even character-level representations cannot recover from such errors. Similarly sounding words may give very dissimilar word embeddings. We argue that representing text as a sequence of phoneme embeddings can help when dealing with ASR errors.
We propose to represent ASR transcriptions as sequences of phonemes. We map phonemes to phoneme embeddings and propose several methods to train phoneme embeddings that are able to capture pronunciation similarities. Finally, we use these pre-trained embeddings as inputs to Neural Network architectures for solving NLU tasks.
To learn phoneme embeddings we design phoneme2vec. We propose 4 variants listed below (p2vc, p2vm, and p2va, s2s). These training procedures for learning phoneme embeddings require a corpus of corresponding REF and ASR utterances. We propose an automatic data generation pipeline (fig. 5). Using this pipeline we created noisy versions of four NLU datasets that we plan to make available to the community.
1) p2vc: phoneme2vec on surrounding phonemes. Given a phoneme we want to predict its surrounding phonemes with the traditional word2vec procedure. We decided not to limit the context of a phoneme to its word as the ASR might have failed the word segmentation. However, we explicitly consider the padding symbol (fig. 1) as it represents the "absence of sound". We aim to capture pronunciation similarities. Intuitively, two phonemes are similar if the ASR often confuses them. Following this intuition we further propose two variants of phoneme2vec.
2) p2vm: phoneme2vec on mixed REF and ASR utterances. We mix REF and ASR utterances at phoneme level in an alternating way (fig. 2).
3) p2va: phoneme2vec on aligned REF and ASR utterances. This is a more general solution that involves an explicit alignment. We directly pair phonemes in a REF utterance with their aligned phonemes in the ASR utterance using the Needleman-Wunsch alignment algorithm (fig. 3).
4) s2s. Another very intuitive way of training phoneme embeddings is to use a seq2seq model. After reading the entire REF utterance, the last hidden state of the LSTM is passed to decoder (fig. 4). The targets are ASR utterance. So, we train the seq2seq model to predict the next correct phoneme of the ASR utterance given the REF utterance and the previous ASR phonemes. In this sequence-to-sequence architecture, we add phoneme embedding layers before the encoder and decoder; so, utterances are transformed into sequences of phonemes that are given as inputs to the encoder and decoder. Finally, the embedding layer of the decoder is used as pre-trained phoneme embeddings.
Varying hyperparameters, overall, we create 10 different pre-trained phoneme embeddings. We set the dimension size of the embedding vector to 20 for all the methods. Analysis suggests that p2vc is not suited for learning pronunciation aspects (fig. 7), while the other proposed models more effectively capture these desired properties (fig. 6).
We show that models exploiting our phoneme representation can significantly improve classification performance on datasets containing ASR errors compared to models operating only on standard character or word representations.
IMO, if we convert ground truth transcription to phonemes, then we obtain "ideal" transcription, where all phonemes are well pronounced. This may not match real spoken words. In this case, it is not clear if phonetic NLU inputs are better than letters. Indeed, phoneme input may represent phonetic uncertainty, but text input also can: if we input "A", we mean "all words that sound similar as A". Maybe, text represented as phonetic embeddings is richer than text of letters, since it contains uncertainty, but this is just because this it is not a result of argmax operation. So, this paper needs ablations: if ASR stage outputs letters or BPE tokens (not phonemes), and we pass output token embeddings into NLU stage and train this stage, will it be worse?


# Phoneme-BERT: Joint Language Modelling of Phoneme Sequence and ASR Transcript
In spoken language understanding (SLU), usually the pipeline consists of ASR and NLU stages. However, ASR errors degrade the performance of the  NLU stage. The approaches tried by scientific community to address the errors: 1) Modelling word confidence in the ASR stage, 2) ASR correction by LM, 3) end-to-end NLU models, 4) Phoneme enhanced representations.
We propose PhonemeBERT that jointly models the phoneme and ASR sequence (fig. 1). We generate phoneme sequence from a separate phonemic listen-attend-spell (LAS) model. The phonemic LAS model is a sequence-to-sequence model with phoneme as its output unit.
IMO, need full reading to understand anything in this paper.


# ByT5 model for massively multilingual grapheme-to-phoneme conversion
G2P (grapheme-to-phoneme conversion) is a fundamental to the pipeline for a variety of speech processing tasks that depend on phonemic inputs, including speech synthesis and speech recognition (IMO, why so?).
To create a training dataset, we aggregated pronunciation dictionaries previously published or made available in around 100 languages.
We have curated a G2P dataset from various sources that covers around 100 languages and trained large-scale multilingual G2P models based on ByT5. It significantly outperformed the token-based mT5 model.
Multilingual models can perform zero-shot G2P on unseen low-resource languages with seen writing systems (IMO not clear how to)
See examples in https://github.com/lingjzhu/CharsiuG2P


# State-of-the-art Speech Recognition With Sequence-to-Sequence Models
To date, none of end-to-end ASR models (LAS, RNN-T, Neural Transducer, Monotonic Alignments, RNA) has been able to outperform a SOTA conventional systems on a large vocabulary continuous ASR.
We explore a variety of improvements to the LAS model, since previous work showed that LAS offered improvements over other seq2seq models.
Word piece models can be used instead of graphemes, giving a modest improvement.
Multi-head attention architecture, which allows the model to learn to attend to multiple locations of the encoded features, offers improvements over the single-head attention.
Minimum WER (MWER) optimization significantly improves performance. The loss function can then be approximated using the set of N-best hypotheses computed using beam-search decoding.
We include scheduled sampling (SS), which feeds the previous label prediction during training rather than ground truth.
Label smoothing helps to make the model less confident in its predictions.
Synchronous SGD offer improvements over asynchronous SGD.
Overall, we get 13% relative improvement in WER with structure improvements (word piece, multi-head attention) and 27.5% relative improvement with optimization strategies (MWER, label smoothing, SS, synchronous SGD).
Finally, we incorporate a LM to rescore N-best lists in the second pass (eq. 3), which results in a further 3.4% relative improvement in WER (IMO very modest improvement). The external LM is a large 5-gram LM trained on text data from a variety of domains. Domain-specific LMs are first trained, then combined together using Bayesian-interpolation.
We also present results with a unidirectional LSTM encoder for streaming recognition.
The training utterances are anonymized and hand-transcribed, and are representative of Google’s voice search traffic. This data set is created by artificially corrupting clean utterances using a room simulator, adding varying degrees of noise and reverberation such that the overall SNR is between 0dB and 30dB, with an average SNR of 12dB. The noise sources are from YouTube and daily life noisy environmental recordings.


# Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
In seq2seq models, during inference, true previous target tokens are unavailable, and are thus replaced by tokens generated by the model itself, yielding a discrepancy between how the model is used at training and inference. If a wrong decision is taken at time t−1, the model can be in a part of the state space that is very different from those visited from the training distribution and for which it doesn’t know what to do. Worse, it can easily lead to cumulative bad decisions - a classic problem in sequential Gibbs sampling type approaches to sampling, where future samples can have no influence on the past.
So, mistakes made early in the sequence generation process are fed as input to the model and can be quickly amplified.
We propose to flip a coin and use the true previous token with probability E, or an estimate coming from the model itself with probability (1-E). The estimate of the model can be obtained by sampling or argmax.
Intuitively, at the beginning of training, sampling from the model would yield a random token since the model is not well trained, which could lead to very slow convergence.
We propose Scheduled Sampling: a curriculum learning approach to gradually force the model to deal with its own mistakes. We thus propose to use a schedule to decrease E during training.
Future work includes back-propagating the errors through the sampling decisions, as well as exploring better sampling strategies including conditioning on some confidence measure from the model itself.


# An Online Sequence-to-Sequence Model Using Partial Conditioning
Seq2seq models are unsuitable for tasks where it is important to produce outputs as the input sequence arrives (online prediction).
We present a Neural Transducer. Unlike sequence-to-sequence models, it computes the next-step distribution conditioned on the partially observed input sequence and the partially generated sequence.
The inputs to the transducer RNN come from two sources: the encoder RNN and its own recurrent state. At each time step, the transducer can decide to emit zero to many output symbols (fig. 1, 2).
During training, alignments of output symbols to the input sequence are unavailable. We show how a dynamic programming algorithm can be used to compute "approximate" best alignments (sec. 3.5).
We achieve SOTA for unidirectional models on the TIMIT phoneme recognition task, using a Neural Transducer with 3 layered unidirectional LSTM encoder and 3 layered unidirectional LSTM transducer.
We find that the Neural Transducer performs well for long sequences even when attention mechanisms are not used.
IMO, such a method does not allow to disambiguate previous transcriptions when new words arrive, so this will give strictly lower quality than conditioning on the full inputs.


# Online and Linear-Time Attention by Enforcing Monotonic Alignments
Seq2seq RNN requires the model to effectively compress all important information about the input sequence into a single vector. In practice, this often results in the model having difficulty generalizing to longer sequences than those seen during training. An effective solution to these shortcomings are attention mechanisms. Similar mechanisms have been used as soft addressing schemes in memory-augmented NN architectures. A common criticism of soft attention is that the model must perform a pass over the entire input sequence when producing each element of the output sequence. This gives quadratic time complexity, and also soft attention cannot be used in online prediction, when the input is only partially observed.
In seq2seq tasks, the alignment between input and output sequence elements is roughly monotonic in many problems of interest (sentence summarization, NMT, ASR). Also, in neural networks, in many cases the attention is assigned mostly to a single entry.
In standard attention, eqs. (2) and (3) are computing the expected output of a simple stochastic process, when a memory index is sampled from a categorical distribution.
We then formulate a stochastic process which explicitly processes the memory in a left-to-right manner. Our novel process can be computed in an online manner; i.e. we do not need to wait to observe the entire input sequence before we start producing the output sequence. (IMO this means that the encoder should be autoregressive?)
We then propose training with respect to the expected value of the described online sampling process. Out training algorithm still has a quadratic complexity, but it allows linear-time attention process at test time.
We need our mechanism to encouraging discreteness to exhibit similar behavior when training in expectation and when using the hard monotonic attention process at test time. A straightforward way to encourage this behavior is to add noise before the sigmoid. This approach is similar to the recently proposed Gumbel-Softmax trick, except we did not find it necessary to anneal the temperature.
IMO, at test time the information flow is bottlenecked since the decoder can only attend to one encoder state (fig. 3).


# Recurrent Neural Aligner: An Encoder-Decoder Neural Network Model for Sequence to Sequence Mapping
We propose an encoder-decoder RNN called Recurrent Neural Aligner (RNA).
Like CTC models, it defines a probability distribution over target label sequences including blank labels corresponding to each time step in input. The probability of a label sequence is calculated by marginalizing over all possible blank label positions. Unlike CTC, RNA does not make a conditional independence assumption for label predictions.
RNA is capable of streaming recognition since the decoder does not employ attention mechanism.
RNA grapheme model with greedy search and no external language model can match the accuracy of a CTC word model which uses a word n-gram language model for rescoring when we have large amount of acoustic training data with transcripts available.


# Improving Attention Based Sequence-to-Sequence Models for End-to-End English Conversational Speech Recognition
Current end-to-end ASR systems still cannot achieve a comparable performance to a conventional ASR system on the Switchboard dataset, a widely used English conversational speech benchmark.
We propose to use the input-feeding architecture which feeds not only the previous context vector but also the previous decoder hidden state information as inputs (fig. 1, baseline is similar to LAS).
We propose a better hypothesis generation scheme for sequential minimum Bayes risk (MBR) training. We use a simple beam-search algorithm to generate the hypothesis set and rescore with eq. 14. We’ve also tried the attention coverage penalty for re-scoring but it never worked in our experiment. Seq2seq model tends to make over-confident predictions. For beam-search, over-confident predictions will lead to too many alike hypothesized sequences among N-best which might prevent the MBR training procedure from seeing a more diverse hypothesis space. To this end, we introduce softmax smoothing during N-best generation, using softmax temperature (eq. 15). (IMO there are similar works, like "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models").


# Optimal Completion Distillation for Sequence Learning
Maximum Likelihood Estimation (MLE) is still considered the dominant approach for training seq2seq models. Alternative approaches typically do not offer a substantial performance improvement over a well tuned MLE baseline, especially when label smoothing and scheduled sampling are used.
We present Optimal Completion Distillation (OCD), a training procedure for optimizing sequence to sequence models based on edit distance.
In OCD, we always train on prefixes generated by sampling from the model that is being optimized. For each generated prefix, we identify all of the optimal suffixes that result in a minimum total edit distance v.s. the ground truth target using an efficient dynamic programming algorithm. We then maximize the average log probability of the first token of each optimal suffix (table 1).
Generally, OCD excels at training from scratch, which makes it an ideal substitution for MLE. Hence, OCD is orthogonal to methods which require MLE pretraining or joint optimization.
Assuming that the generated prefix sequence is perfectly matched with the ground truth sequence, then the OCD targets would simply be the following tokens of the ground truth sequence. Hence, OCD becomes equivalent to MLE. However, during training, the generated prefixes sampled from the model do not match the ground truth sequence, even at the end of training. This suggests that OCD and MLE are training on very different input prefix trajectories.
On both WSJ and Librispeech, our proposed OCD algorithm significantly outperforms our own strong baselines including MLE (Maximum Likelihood Estimation with label smoothing) and SS (scheduled sampling with a well-tuned schedule). Even at the same training CER, we observe better validation error for OCD, which suggests that OCD improves generalization of MLE, possibly because OCD alleviates the mismatch between training and inference.
IMO, this may be implemented in BPE decoders either on token level, or on character level. This may be similar to training on random tokenizations, instead of the deterministic BPE.


# Sequence-to-Sequence Learning as Beam-Search Optimization
There are major, previously known issues with seq2seq models:
1) Exposure Bias: the model is never exposed to its own errors during training (this was also addressed in "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks").
2) Loss-Evaluation Mismatch: training uses a word-level loss, while at test-time we target improving sequence-level evaluation metrics, such as BLEU.
3) Label bias: word probabilities at each time-step are locally normalized, guaranteeing that successors of incorrect histories receive the same mass as do the successors of the true history (see "Globally Normalized Transition-Based Neural Networks").
We define a loss function in terms of errors made during beam search. We provide an efficient algorithm to backpropagate through the beam-search procedure during seq2seq training.
We learn to produce (non-probabilistic) scores for ranking sequences. We do not use softmax, thereby allowing the model to avoid issues associated with the label bias problem. Ideally we would train by comparing the gold sequence to the highest-scoring complete sequence. However, finding the argmax sequence according to this model is intractable. We propose to adopt a LaSO-like scheme to train, which we will refer to as beam search optimization. We define a loss that penalizes the gold sequence falling off the beam during training (fig. 1).
We run experiments on three very different problems: word ordering, syntactic parsing, and machine translation. The version with beam search optimization shows significant improvements on all three tasks, compared to a highly tuned seq2seq system with attention.


# The PyTorch-Kaldi Speech Recognition Toolkit
Kaldi currently represents the most popular ASR toolkit. It relies on finite-state transducers (FSTs) and provides a set of C++ libraries.
Our PyTorch-Kaldi project aims to bridge the gap between Kaldi and PyTorch. It implements acoustic models in PyTorch, while feature extraction, label/alignment computation, and decoding are performed with Kaldi, making it suitable to develop SOTA DNN-HMM speech recognizers (fig. 1).


# Token-Level Ensemble Distillation for Grapheme-to-Phoneme Conversion
We propose the token-level ensemble distillation for grapheme-to-phoneme (G2P) conversion task. Specifically, we train a teacher model to generate the phoneme sequence as well as its probability distribution given unlabeled grapheme sequence, and regard the unlabeled grapheme sequence and the generated phoneme sequence as pseudo labeled data, and add them into the original training  data.
We train a variety of models (CNN, RNN and Transformer) for ensemble to get higher accuracy, and transfer the knowledge of the ensemble models to a light-weight model.


# Deep context: end-to-end contextual speech recognition
Speech recognition performance can be improved by incorporating information about the speaker’s context, such as the dialog state, the speaker’s location, personalized information about the user.
We propose Contextual-LAS (CLAS) for ASR task, which can leverage a list of contextual phrases to improve recognition performance.
Our technique consists of first embedding each phrase, represented as a sequence of graphemes, into a fixed-dimensional representation, and then employing an attention mechanism to summarize the available context at each step of the model’s output predictions (fig. 1).
Our method does not require careful tuning of rescoring weights, while still being able to incorporate out-of-vocabulary (OOV) terms.


# Joint Grapheme and Phoneme Embeddings for Contextual End-to-End ASR
We improve over the CLAS approach for contextual ASR.
In CLAS, because the embeddings are learned from graphemic information only, they do not discriminate well among similar sequences of graphemes, nor do they generalize well to unseen pronunciations of words. (IMO not clear why pronunciations matter here, since context should be used on the language modeling internal stage, not on the acoustic stage).
To overcome these problems, 1) we extract embeddings based on a grapheme-to-phoneme (G2P) encoder-decoder, and 2) we try to leverage the power of the bidirectional LSTM instead of LSTM proposed in CLAS.


# One Model to Pronounce Them All: Multilingual Grapheme-to-Phoneme Conversion With a Transformer Ensemble
We are participating in the SIGMORPHON 2020 Shared Task, when sequences of graphemes are mapped to corresponding phonemes. A clear challenge is the limited size of the shared task training data for each of the 15 individual languages.
We build a single transformer model mapping from input to output across all the languages simultaneously. We also use pseudo-labeling on Wikipedia data, selecting sequences of phonemes predicted with our models above a certain confidence threshold.


# The SIGMORPHON 2020 Shared Task on Multilingual Grapheme-to-Phoneme Conversion
ASR requires mappings between written words and their pronunciations, either explicit or implicit (in end-to-end models). For open-vocabulary applications, these mappings must generalize to unseen words, and so must be expressed as mappings between sequences of graphemes and phonemes or phones. We note that the term phoneme is a well-defined object in linguistic theory, which may not be appropriate for a given pronunciation dictionary. Therefore, in what follows we use the term phone to refer to transcriptions symbols.
Rule-based systems require linguistic expertise to develop and maintain, and may be brittle or inaccurate. Therefore, modern speech engines usually treat grapheme-to-phoneme conversion as a machine learning problem (IMO the authors still do not explain why we need this in end-to-end systems).
The vast majority of published research focuses on English or a few other highly-resourced, globally hegemonic languages for which free pronunciation dictionaries are available.
We present SIGMORPHON 2020 shared task: a multilingual grapheme-to-phoneme conversion task with data sets, evaluation metrics, and strong baselines. The task included data from 15 languages and scripts. 9 teams submitted 23 G2P systems and achieved substantial improvements over the provided baselines.


# End-to-end ASR to jointly predict transcriptions and linguistic annotations
Our primary goal is to provide aligned transcripts (phonemes) and linguistic annotations (graphemes) with minimal degradation in ASR performance. Given these outputs, we can easily combine ASR with downstream NLP tasks and also conduct an intuitive error analysis (e.g., detecting the error caused due to the homonym by checking the word and the corresponding phoneme output).
We consider several existing options to make the ASR system predict both phonemes and graphemes (fig. 1). The options A and B do not align both output sequences. The third option, in contrast, does not require postprocessing to align the label sequences.
We choose the option C and adopt a model to output phonemic transcripts and POS tags, as well as graphemes (fig. 2f).
Our approach predicts linguistic annotations correctly even though corresponding graphemes are wrong. This feature is helpful for the downstream NLP system like slot filling or intent detection.


# Improving Transformer-Based End-to-End Speech Recognition with Connectionist Temporal Classification and Language Model Integration
We address two problems with Transformer end-to-end ASR (see "Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition"):
1) Slower convergence, namely its slower increase in validation accuracy over wall clock time, than RNN-based ASR. Transformer takes less time per iteration, but it takes many more epochs to converge (fig. 1).
2) The difficulty of LM integration in joint beam search decoding. The scores provided by Transformer and LM had drastically different behaviours that make them difficult to combine.
We implement CTC joint training as a multi-task learning by adding a new branch from Transformer encoder. It can make the convergence of the seq2seq model faster because CTC learns to align the speech feature and transcription explicitly.
We found that the joint decoding with CTC could perform better with LM integration than the one without CTC in our experiments. We followed the common joint decoding approach, which simply takes the sum of log probabilities from the CTC, decoder and LM (eq. 15).


# XPhoneBERT: A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech
In current TTS systems, the pre-trained BERT is used to provide additional contextual information and helps increase the quality of the output synthesized speech (for example, see "Pre-trained Text Embeddings for Enhanced Text-to-Speech Synthesis"). Therefore, it might be better if the contextualized phoneme representations are directly produced by a pre-trained BERT-type model that is learned from unlabeled phoneme-level data.
Among recent works, these are PnG BERT (takes both phonemes and graphemes as the input), Mixed-Phoneme BERT (takes both phonemes and sup-phoneme tokens as the input), Phoneme-level BERT (only taking phonemes as the input and employs an additional auxiliary task that predicts the corresponding grapheme for each phoneme).
It is worth exploring pre-trained models for phoneme representations in languages other than English.
We present and publicly release XPhoneBERT trained using the RoBERTa pre-training approach on 330M phoneme-level sentences from nearly 100 languages and locales. To convert texts into phonemes, we employ the CharsiuG2P toolkit that supports 90+ languages and locales.
Employing XPhoneBERT as an input phoneme encoder significantly boosts the performance of a strong TTS model in terms of naturalness and prosody.


# SoundChoice: Grapheme-to-Phoneme Models with Semantic Disambiguation
Popular end-to-end speech synthesis models often fail to perform disambiguation of the homographs - a sequence of graphemes that can yield different pronunciations depending on the context (e.g. "read" - past vs present).
Grapheme-to-Phoneme (G2P) models can improve the system’s performance in these cases. However, these models are typically trained and evaluated on word-level lexicons (e.g., CMUDict), making it impossible to resolve homograph disambiguation.
We propose SoundChoice, a novel G2P model that operates at the sentence level. It enables the model to exploit the context and better resolve homograph disambiguation. SoundChoice models the sentence context using mixed representation composed of characters and BERT word embeddings (fig. 1).
SoundChoice uses CTC loss on top of the encoder and the standard sequence-to-sequence loss computed after the decoder. To further improve disambiguation, we propose a homograph loss that penalizes errors made on homograph words.
SoundChoice gradually switches from word- to sentence-level G2P using a curriculum learning strategy.
We also release the new LibriG2P dataset that combines data from LibriSpeech Alignments and the Wikipedia Homograph.


# Mixed Orthographic/Phonemic Language Modeling: Beyond Orthographically Restricted Transformers (BORT)
Explicit representation of phonology (such as processing ambiguous and noisy output from ASR systems, or handling of names and neologisms) are under-served by the current pre-training paradigm. LLM support for the international phonetic alphabet (IPA) ranges from poor to absent. The data used to pre-train an LLM incidentally contains little IPA content if any at all. Also, a task like MLM has little to gain from learning the sound relationships between words, so we have no reason to expect these models to adapt to phonetic tasks as well as they do semantic ones.
We propose BORT (Beyond Orthographically Restricted Transformers) by extending the pre-training of an existing LLM, BART. Given a document, we transform some words into IPA, then train the model to restore the orthography.
We evaluate the utility of BORT by fine-tuning to two clinically-motivated tasks. The model learned the task that the human annotators performed in AphasiaBank. In "hard" variant, we train the model to fill in paraphasias (i.e., incorrect pronunciations) with the intended orthographic word (given the surrounding context).


# SCRAPS: Speech Contrastive Representations of Acoustic and Phonetic Spaces
On the speech generation side, one of the main difficulties is to build a model that correctly aligns the phonetic and acoustic sequences, leading to a natural prosody with fluent speech and high intelligibility. On the opposite side, ASR systems struggle with long-tail words recognition, and speech vs background disentanglement.
We propose SCRAPS (Speech Contrastive Representation of Acoustic and Phonetic Spaces) (fig. 1). To the best of our knowledge, our work is the first attempt to use a CLIP-based strategy to learn joint phonetic and acoustic spaces.
For phonetic encoder, the input sequences are derived from the text transcriptions by using a simplified grapheme to phoneme processor (G2P) based on a dictionary of pronunciations.
In our model, LSTMs of the two encoders share weights to propagate the latent spaces compatibility back to the output of the transformer outputs (we empirically prove this assumption). That helps the time-dependent vectors live in the same space for both encoders. We use a contrastive loss to maximize the scores of the matching pairs, while minimizing the scores of the non-matching pairs.
We have trained a SCRAPS model on a large proprietary dataset composed of recordings of untrained speakers with variety of background noises, unnormalized pauses, and in some cases, even some samples with concurrent speech where one of the speakers dominates over the others. Each recording is accompanied with its corresponding transcription.
We perform a sensitivity and robustness analysis to study how does the model react against perturbation of the input data.
Downstream application 1: pretrained phonetic embeddings for speech generation. We have trained an autoregressive multispeaker text-to-speech model with attention on a proprietary dataset with 180,000 hours of de-identified en-US recordings. The acoustic decoder is autoregressive and attends to the output of the phonetic encoder and the speaker embedding. Then we have substituted the phonetic encoder by a SCRAPS pretrained phonetic encoder (only the transformer backbone, not the LSTM integrator). We observe that both architectures get a very similar final performance, but when using SCRAPS, the model converges much faster.
Downstream application 2: text-less intelligibility evaluation for voice conversion systems. Although SCRAPS is trained to match a sequence of phonemes to the corresponding audio, at inference time it can also be used to compute correspondence between two audio files without requiring any text. In this scenario, the SCRAPS score is computed between vectors of synthetic audio (VC) and source audio (pre-conversion).
IMO, it could be possible to use dynamic matching for both backbone output vector sequences, alleviating the need to compress all vector sequence in a single vector, which seems unreasonable.


# Improving noise robust automatic speech recognition with single-channel time-domain enhancement network
There is a need for more research on effective single-channel speech enhancement (SE) front-ends for ASR. Although frequency masking approaches have successfully improved SE evaluation metrics, e.g., signal-todistortion ratio (SDR), this improvement did not lead to better ASR performance. This suggests that most single-channel SE approaches tend to introduce distortions that create a mismatch with the ASR back-end, therefore limiting their effect on ASR. Time-domain approaches have not been sufficiently investigated in the context of noise-robust ASR.
We adapt Conv-TasNet for the noise reduction task and call it Denoising-TasNet. We investigate two variants of Denoising-TasNet, one predicting only the enhanced speech and one with two outputs predicting speech and noise. The latter enables defining a multi-task loss, which can regularize the network training and is shown to achieve better ASR performance.
We perform experiments on CHiME-4 data. Denoising-TasNet significantly reduces WER on real recordings. It can improve ASR performance even without retraining the ASR back-end.
These demonstrates that single-channel noise reduction can still improve ASR performance.


# Optimizing Two-Pass Cross-Lingual Transfer Learning: Phoneme Recognition and Phoneme to Grapheme Translation
We propose two-pass ASR system that first performs phoneme recognition and then translates the recognized phonemes into graphemes. Our methodology aims to advance ASR systems’ effectiveness in low-resource languages.
Rather than relying on grapheme units, which may not leverage the advantages of cross-lingual transfer learning, we choose to utilize phoneme units as the final output of our ASR system. We incorporate a dedicated translation model that converts phoneme outputs into grapheme units.
The IPA phoneme representation may not efficiently capture the shared phonetic characteristics across languages. Moreover, phoneme recognition results can be inaccurate, leading to error propagation during the phoneme-to-grapheme translation step.
We propose a novel approach called Pivot Phoneme Merging (PPM) to address these challenges. It groups phonemes based on shared articulatory features, facilitating improved vocabulary sharing across languages.
We also present a Global Phoneme Noise (GPN) generator that enables the pseudo-labeling of external text corpora, incorporating realistic ASR noise into the training process for P2G translation.


# Phoneme-aware Encoding for Prefix-tree-based Contextual ASR
End-to-end ASR systems have difficulties in recognizing uncommon words. Contextual biasing is a method to incorporate a contextual knowledge. We pass the model a list of words that are likely to appear in the context.
1) One approach is shallow fusion with a contextual LM, where biasing words are compiled into Weight Finite State Transducer. However, they require some heuristics and careful tuning of an LM weight to avoid under- or over-biasing.
2) In attention-based deep context approaches, each biasing word is converted into an encoding vector, and an ASR decoder attends to the encodings (word-level biasing). However, they have an issue handling a large number of biasing words.
3) To efficiently handle them, a prefix tree, or a trie, -based deep biasing methods have been considered (subword-level biasing).
4) TCPGen further extends prefix-tree-based biasing. It works with a pre-trained ASR model such as Whisper without modifying their architecture.
Existing prefix-tree-based biasing methods rely solely on their textual representations. As rare biasing words sometimes have pronunciations that are difficult to estimate from text, it is important to provide their pronunciation information as a clue to recognize such words. This is especially common for ideographic characters such as Japanese kanji. For subword-level biasing, phonemes aligned to each subword are required. This is not trivial because pronunciation is typically defined for the entire word.
We propose subword-level phoneme-aware encodings for TCPGen. To obtain alignment from a subword to phonemes, we consider using the attention weights of seq2seq G2P model or EM algorithm-based alignment. It would be preferable for queries to be also explicitly aware of phonemes. To this end, we train the end-to-end ASR model with auxiliary CTC loss whose target is a phoneme sequence, and the CTC predictions are incorporated into the formulation of query in TCPGen.


# Whistle: Data-Efficient Multilingual and Crosslingual Speech Recognition via Weakly Phonetic Supervision
In multilingual and crosslingual ASR (MCL-ASR), while requiring pronunciation lexicons, pre-training with phonetic supervision is more advantageous for information sharing between different languages. There have been no solid experiments to study which approach is better or if they yields similar results.
Phoneme-based models naturally overcome language imbalance and can be efficiently trained on natural data mixing, while subword-based models need careful tokenization and data mixing in training.
We propose Whistle (Weakly phonetic supervision strategy for multilingual and crosslingual speech recognition) to explore supervised pre-training with weakly phonetic supervision, towards data-efficient MCL-ASR (this is in spirit similar to weakly graphemic supervision in Whisper). We obtain the IPA phonetic transcripts by leveraging the LanguageNet G2P models available for 142 languages with the phoneme error rates (PERs) ranging from 7% to 45%.
Besides the performance advantage of phoneme-based supervision over subword-based supervision, we find that phoneme-based models tend to be more training efficient, i.e., they can converge with fewer optimzation steps, with 24% reduction.


# Streaming Small-Footprint Keyword Spotting using Sequence-to-Sequence Models
Keyword spotting is the task of detecting specific words or phrases in speech utterances. An example of such technology is speech-enabled assistant “Okay/Hey Google” on Google Home.
We explore RNN-T, to build a streaming keyword spotting system which can be used to detect arbitrary keywords.
We find that RNN-T system trained to predict phonemes, when augmented with an additional "end-of-word" symbol strongly outperforms a strong keyword-filler baseline (fig. 4). The <eow> token is useful, for example, to prevent detection of the keyword "Erica (E r\ @ k @)" inside the word "America (@ m E r\ @ k @)" Additionally, we propose a novel technique to bias the search towards a specific keyword of interest using an attention mechanism (fig. 1c).


# Cold Fusion: Training Seq2Seq Models Together with Language Models
Because LMs can be trained from abundantly available unsupervised text corpora, they can improve Seq2Seq’s performance. A standard way is to linearly combine the score of the task-specific Seq2Seq model with that of an auxiliary langauge model to guide beam search (Shallow Fusion).
Deep Fusion (see "On using monolingual corpora in neural machine translation") learns to fuse the hidden states of the Seq2Seq decoder and a neural LM with a gating mechanism, after the two models are trained independently.
The biggest disadvantage with Deep Fusion is that the task-specific model is trained independently from the LM. This means that the Seq2Seq decoder needs to learn a LM from the training data labels, which can be rather parsimonious compared to the large text corpora available for language model training. For example, if a Seq2Seq model fully trained on legal documents is later fused with a medical language model, the decoder still has an inherent tendency to follow the linguistic structure found in legal text. Thus, in order to adapt to novel domains, Deep Fusion must first learn to discount the implicit knowledge of the language.
Also, in Deep Fusion a considerable portion of the decoder capacity is wasted, since decoder learns an implicit language model.
We introduce Cold Fusion to overcome both these limitations. Cold Fusion encourages the Seq2Seq decoder to learn to use the external language model during training.
Cold Fusion can almost completely transfer to a new domain for the speech recognition task with 10 times less data.
At each time step t, we obtain LM output probabilities and process them with DNN (eq. 4a). Then we concatenate the state of the task specific model with the DNN outputs, with gating (eq. 4b, 4c). We then process the result with another DNN (eq. 4d), obtaining the final probabilities. Since we use LM probabilities, we can generalize to new LMs (with the same vocabulary) at inference time. Since LM logits can have arbitrary offsets, the maximum value is subtracted off before feeding into the layer. While DNNs can have any depth, we found a single affine layer with ReLU to be helpful.
We collected two data sets with audio recordings: one based on search queries which served as our source domain, and another based on movie transcripts which served as our target domain.


# Shallow-Fusion End-to-End Contextual Biasing
In ASR, contextual biasing to a specific domain, including a user’s song names, app names and contact names, is an important component of any production-level system.
Compared to conventional models, end-to-end models make more errors in rare, context-dependent words and phrases. Also, proper nouns are usually pruned during beam search before contextual biasing can be applied, as biasing happens at the end of a word (rather than the grapheme/wordpiece units the E2E model predicts).
We explore biasing at the sub-word unit level (grapheme, wordpiece) rather than the word-level.
Second, we explore applying the contextual finite state transducer (FST) before beam pruning rather than after.
Third, because contextual n-grams are typically used with a common set of prefixes ("call", "text"), we investigate incorporating these prefixes into shallow fusion. This helps tremendously to avoid degradation for anti-context (on utterances we do not want to bias).
Fourth, by improving proper noun modeling by training with a large amount of unsupervised data, we can improve performance further.


# On Using Monolingual Corpora in Neural Machine Translation
Incorporating monolingual corpora can improve a translation system on a low-resource, as well as high-resource language pair, and a domain restricted translation problem (Chinese-English SMS chat).
We propose two ways to integrate a LM trained only on monolingual data (target language) into an NMT system: shallow fusion and deep fusion.
Both models are word-based: vocabularies are constructed with the most common words in the parallel corpora (sec. 6.1.1, 6.2.3).
Without loss of generality, we use a LM based on RNN.
Shallow fusion is analogous to how LMs are used in the decoder of a usual statistical machine translation system. At each time step, we have a set of M hypotheses (beams?) and N possible next words for each hypothesis. This allows us to calculate MN scores: each score is the summation of the score of the hypothesis and the score given by the NMT to the next word. These scores are then sorted, and top K ones are selected as candidates. Then the candidates are rescored with LM by weighted sum of the scores given by the translation model and the language model, where weight is a hyper-parameter (eq. 5).
In deep fusion, we concatenate hidden states of NMT decoder and LM. The model is then finetuned to use the hidden states from both of these models when computing the output probability of the next word. In this paper, we tune only the output parameters to ensure that the structure learned by the LM from monolingual corpora is not overwritten.
In deep fusion, we also augment the decoder with a "controller" mechanism. For example, if a noun is to be translated, it may be better to ignore any signal from the LM, as it may prevent the decoder from choosing the correct translation. Intuitively, this mechanism helps the model dynamically weight the different models depending on the word being translated. At each time step, the controller takes the hidden state of the LM as input and outputs a scalar. It is then multiplied with the hidden state of the LM.
Where the domain of the bilingual and monolingual corpora were similar (De-En, Cs-En), we observed improvement with both deep and shallow fusion methods. In the case where they were dissimilar (Zh-En), the improvement using shallow fusion were much smaller.


# Deep context: end-to-end contextual speech recognition
We propose Contextual-LAS (CLAS), all-neural mechanism which can leverage contextual information – provided as a list of contextual phrases – to improve ASR performance.
Our technique consists of first embedding each phrase into a fixed-dimensional representation, and then employing an attention mechanism to summarize the available context at each step of the model’s output predictions (fig. 1b).
The proposed method does not require that the particular context information be available at training time, and the method does not require careful tuning of rescoring weights.
This is a generalization of the previously proposed method in "Streaming small-footprint keyword spotting using sequence-to-sequence models".


# Audio-attention discriminative language model for ASR rescoring
We proposed to use an attention-based LM for second-pass rescoring of N-best lists generated by a conventional ASR system.
An RNNLM style model is trained using word-level contextual input, while simultaneously attending to audio, using a minimum word error rate criterion, which learns to rescore the N-best hypotheses list from a first pass system.


# A Density Ratio Approach to Language Model Fusion in End-To-End Automatic Speech Recognition
There are many situations where we would like to use a separate LM to complement or modify a given ASR system to 1) make use of text-only training data, 2) bias the recognition grammar towards a list of specific words or phrases for a specific context.
Approaches such as Deep Fusion, Cold Fusion and Component Fusion have not replaced the simple Shallow Fusion method as the go-to method in most of the ASR community, because Shallow Fusion does not require model retraining.
Sec. 2 describes the Noisy Channel generative model underlying the origins of statistical ASR. It combines generative acoustic and language models. Though lacking in discriminative power, the paradigm provides a clear theoretical framework for decoupling the acoustic model p(X|W) and LM p(W) (where X is audio and W is text). Also this section describes the "hybrid" approach that allows to estimate a "pseudo-generative" score with a NN and then use eq. 1. In comparison, the popular Shallow Fusion approach is not justified according to probability theory.
We propose Density Ratio method which produces consistent gains over Shallow Fusion in a cross-domain scenario.
Let we have a source domain ASR model p_source(W|X), called a "posterior", a source domain LM p_source(W), a target domain LM p_target(W), and let p(X|W) be equal in both domains: the domains are acoustically consistent.
We can estimate a target posterior with eq. 6, where we call p_target(W) / p_source(W) a "density ratio". We can inject this term to RNN-T decoding with eq. 11.
So, our method is purely a decode-time method, no joint training is involved, but it does require tuning of the LM scaling factor(s), as does Shallow Fusion. A held-out set can be used for that purpose.
IMO, in eq. 1 it is not clear how do we obtain the final prediction; also in eq. 1, while there is no length penalty, it is not really clear what does p(text) means; usually LMs include <eos> token that has no connection with real life; for example, what is larger: the probability of all texts of length 10 or the probability of all texts of length 20? a problem is that LMs do not model the presence vs absence of text.


# Integrating Knowledge into End-to-End Speech Recognition from External Text-Only Data
In end-to-end ASR is worth to investigate a different method to integrate the knowledge from the text-only data without external modules (LMs) at the inference stage. Another issue is that autoregressive decoders are difficult to leverage the right context.
We propose Learn Spelling from Teachers (LST): transferring the knowledge from the LM to the attention-based encoder-decoder via teacher-student learning. The LM  provides soft labels of training transcriptions which are used to train the "student" model.
We propose a LM called Causal clOze completeR (COR), which models the whole context of a sentence. It uses the whole context (including the left context and the right context).
This is an extension of the paper "Learn Spelling from Teachers: Transferring Knowledge from Language Models to Sequence-to-Sequence Speech Recognition".


# Maximum-a-Posteriori-Based Decoding for End-to-End Acoustic Models
In end-to-end ASR, "external" LM is still essential to obtain the best results during the decoding stage although an "internal" LM is implicitly trained in the end-to-end acoustic modeling, because the transcriptions in a speech corpus are normally insufficient for training high quality LMs (CTC has limited representation ability because of the independence assumption in the model; however, even in such case, we think the model somehow learns an internal LM as long as the training criterion of the model is based on the distribution of subword sequences).
How can we integrate the word-level LM score into the subword-level end-to-end ASR? Previous approaches, such as log-linear interpolation, lack theoretical justification and hence has led the systems to produce inconsistent results.
IMO, the formulas in the proposed method are strange, since the transition from eq. 8 to 9 is not clear: 1) why ":=" ? 2) why do we need alpha? 3) if there is a one-to-one mapping between s ans W, why do we need a term Pr(W|s)? It seems like only one term in the sum will be non-zero.
We have previously published two papers on the preliminary work: "Training data pseudo-shuffling and direct decoding framework for recurrent neural network based acoustic modeling" and "Maximum a posteriori Based Decoding for CTC Acoustic Models". Due to the lack of sufficient experiments in various conditions, our previous study had inadequate insights to the framework.


# Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong General Audio Event Taggers
We show (fig. 1) the intermediate representations of Whisper lead to the best linear probing environmental sound classification accuracy, comparing to other model checkpoints such as Hubert, Wav2vec2 etc., indicating Whisper encodes most background sound information.
In addition, for all other ASR models, representations from deeper layers led to lower sound classification accuracies, showing that the models are learning to encode speech information, and ignore background sound information. Whisper does not have this behavior: it encodes background sound information even in its deepest (encoder) layer.
We observe a positive correlation between Whisper’s robustness against a specific background sound type and its potential ability to recognize it. (see fig. 2 and its description, note that the Y axis is inverted). We find the potential ability to recognize a sound type is a necessary but not sufficient condition for Whisper to be robust to it.
It is commonly believed that the representation of a robust ASR model should be noise-invariant. However, the above results reveals that the robustness mechanism of Whisper is different from other ASR models: Whisper first encodes the background sound and then transcribes text conditioned on the type of noise.
Based on the results, we build a unified model for ASR and Audio Tagging (i.e., recognize general audio events).
IMO, 1) two types of background noise that sound very similar are two different classes, the F1 recognition score (fig. 2, X axis) will drop, comparing to both types merged in one class, but Y axis (robustness) will not change. This makes X axis not so much informative. 2) Some types of noise can be more challenging then others with the same SNR. For example, short peaky sounds, even with very high negative SNR, mask only short parts of speech, allowing to restore them from context. Also, low-frequency sounds should distract less than sounds with the speech frequency with the same SNR, etc. The fact that the F1 classification score based on the Whisper encoder outputs for group A is better can be explained by the fact that such noises are more difficult to filter out than monotonous noises. That is, monotonous noise can be filtered out and does not get into the last layer of the encoder by itself, but it is also difficult to isolate the phonemes of the model if this noise is strong, which is why the model's stability to such noise is poor. That is, it is possible that Whisper does not filter out noises from group A not because they are useful in some way (as the authors assume), but because it simply cannot. Although, on the other hand, some noises can of course be useful because they allow us to understand the context in which the speech is made, and therefore the vocabulary used. For example, the noise of a fight can mean the use of vocabulary from computer games, etc.


# A Deep Generative Acoustic Model for Compositional Automatic Speech Recognition
Section 1.1 describes the classical compositional ASR approach, when we model p(audio|text), and then we can combine it with p(text) from LM, using Bayes rule. Gaussian Mixture Models (GMMs) used jointly with 1st-order Hidden Markov Models (HMMs) were well-suited to this modular approach, as they directly provide p(audio|text), see eq. 2-4 (W is text, X is audio, S_w is an alighment of text to audio; IMO the formula 2 is strange because of frame independence assumption). Other probabilistic modules such as a pronunciation dictionary can be introduced into the overall chain, again combining with the other components according to Bayes’ rule.
Section 1.2 describes the adaptation of discriminative DNNs into the above scheme. The popular "hybrid" approach converts p(text|audio) to a scaled p(audio|text) using eq. 5-6 (see "Connectionist speech recognition: a hybrid approach"). The overall construction of a sequence-level training objective, converting local frame-level scores from a discriminative model into "generative" scaled likelihoods, only to then plug those into a discriminative sequence training criterion, may seem like a strange hybrid indeed. Nonetheless, this has been a remarkably effective approach, that still constitutes the SOTA (2018).
Section 1.3 describes end-to-end discriminative sequence-level models, such as LAS. However, combination with LMs is not theoretically justified (sec. 1.4).
... TODO


