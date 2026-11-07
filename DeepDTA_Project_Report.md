# DeepDTA: Deep Drug--Target Binding Affinity Prediction

> **Project report and prototype implementation plan**\
> Based on the research paper: Öztürk, H., Özgür, A. and Ozkirimli, E.
> (2018), *DeepDTA: Deep Drug--Target Binding Affinity Prediction*,
> **Bioinformatics**.
-----------------------------------------------------------------------

## 1. Project Overview

Drug discovery involves identifying molecules that interact with
biological targets, commonly proteins. Experimental testing of every
possible drug--target pair is expensive and time-consuming.
Computational prediction can help researchers prioritise promising pairs
for further laboratory investigation.

**DeepDTA** is a deep-learning approach for predicting the binding
affinity between a drug molecule and a target protein. It receives two
inputs:

1.  A representation of a drug molecule, written as a SMILES string.
2.  The amino-acid sequence of a target protein.

The model processes the two inputs through separate convolutional
neural-network (CNN) branches, combines the learned representations and
predicts a numerical binding-affinity score.

The aim of our project is to build a working DeepDTA-based prototype
with a web interface. A user should be able to submit a drug
representation and a protein sequence, request a prediction, and view
the model's output with an explanation of what that output means.

**Important:** The model's prediction is a computational estimate. It is
not proof that a drug will work in a patient, is safe, or will produce a
clinical benefit. Experimental validation remains necessary.

## 2. Biological Background

### 2.1 Why proteins matter in drug discovery

Cells rely on proteins to perform much of their work. Proteins help
control chemical reactions, transmit signals, transport substances,
provide structural support and regulate cellular processes. A protein is
made from a chain of amino acids. The chain folds into a
three-dimensional structure, and that structure helps determine what the
protein does and which molecules it can interact with.

There are 20 commonly occurring standard amino acids in proteins. Each
amino acid has a full name and a one-letter code. A protein sequence is
therefore often represented as a string of these one-letter codes. For
example, a sequence such as `MKWVTF...` is a compact representation of
the amino-acid order in a protein.

The order of amino acids influences how a protein folds and which
chemical and physical properties appear on its surface. Some regions of
a protein can interact with other molecules. These regions are often
called binding sites.

### 2.2 Drugs and their target proteins

Many drugs work by interacting with one or more proteins in the body. In
this context, the drug molecule is often called a **ligand**, and the
protein it interacts with is called the **target**.

A drug may bind to a protein at a particular region. The interaction can
involve several kinds of non-covalent forces, including:

-   **Hydrogen bonding:** attraction involving hydrogen and
    electronegative atoms such as oxygen or nitrogen.
-   **Electrostatic interactions:** attraction or repulsion between
    charged or partially charged regions.
-   **Hydrophobic interactions:** non-polar regions tend to associate
    away from water.
-   **Van der Waals interactions:** weak attractions that arise when
    atoms are close to one another.

A simple lock-and-key analogy can help introduce binding: the drug and
the target must have compatible chemical and spatial properties. Real
proteins, however, are flexible. Their shapes and local environments can
change, so binding is more complex than a rigid lock fitting a rigid
key.

### 2.3 What happens when a drug binds?

Binding can affect a target protein in different ways. A molecule may
reduce the protein's activity, increase its activity, block another
molecule from binding, or alter a signalling process. The biological
outcome depends on the target, the binding location, the molecule and
the wider cellular context.

For example, some cancer treatments target proteins involved in cell
growth and signalling. Kinases are an important family of proteins
because they transfer phosphate groups to other molecules and help
regulate cellular signalling. Abnormal kinase activity can contribute to
disease, which is one reason kinases are studied as drug targets.

Binding alone does not automatically mean that a drug is therapeutically
useful. A molecule can bind strongly and still fail because of
selectivity, toxicity, absorption, distribution, metabolism, excretion
or other biological factors.

### 2.4 Drug--target interaction and binding affinity

A **drug--target interaction (DTI)** describes an interaction between a
drug molecule and a biological target, often a protein. A simple
interaction prediction might classify a pair as interacting or not
interacting. However, this does not describe how strongly the drug
binds.

**Binding affinity** expresses the strength of binding between a ligand
and its target. DeepDTA is designed to predict a numerical
affinity-related value rather than only a yes/no label.

Several experimental quantities are used in binding and inhibition
studies:

-   **Kd (dissociation constant):** describes the equilibrium between
    bound and unbound ligand and target. A lower Kd generally indicates
    tighter binding.
-   **Ki (inhibition constant):** describes the potency of an inhibitor
    in a defined inhibition model. A lower Ki generally indicates
    stronger inhibition under the relevant assumptions.
-   **IC50 (half-maximal inhibitory concentration):** the concentration
    of a compound that produces 50% inhibition in a particular assay. It
    depends on assay conditions and should not be treated as universally
    interchangeable with Kd or Ki.

These quantities are related but are not identical. Their interpretation
depends on the experimental setup and biological system.

### 2.5 pKd and the meaning of a transformed affinity value

The Davis dataset uses Kd measurements. DeepDTA uses a logarithmic
transformation commonly written as:

\[ pK_d = -`\log`{=tex}\_{10}(K_d`\text{ in molar units}`{=tex}) \]

If Kd is supplied in nanomolar (nM), the equivalent expression is:

\[ pK_d = 9 - `\log`{=tex}\_{10}(K_d`\text{ in nM}`{=tex}) \]

Examples:

           Kd   pKd
  ----------- -----
         1 nM     9
        10 nM     8
       100 nM     7
     1,000 nM     6
    10,000 nM     5

A higher pKd corresponds to a lower Kd and therefore generally indicates
stronger binding. The unit conversion is important: the logarithm in the
pKd definition uses Kd in molar units.

### 2.6 The KIBA score

The KIBA dataset combines information from Ki, Kd and IC50 measurements
into a unified score called the **KIBA score**. It was designed to
integrate different types of affinity and inhibition measurements for
modelling.

A KIBA score is not the same thing as pKd. The two outputs have
different definitions and scales. A website must label the selected
dataset and score clearly, and must not convert a KIBA score into a Kd
or pKd unless a scientifically justified conversion is specifically
available.

### 2.7 Why computational prediction is useful

The number of possible drug--protein pairs is very large. Testing every
pair experimentally can require substantial time, cost and laboratory
resources. A prediction model can screen or prioritise pairs so that
researchers can decide which candidates may deserve further
investigation.

The model is therefore a **decision-support and prioritisation tool**,
not a replacement for laboratory experiments, clinical trials or expert
interpretation.

## 3. Research Paper and DeepDTA Method

### 3.1 Core idea

The DeepDTA method represents both the drug and the target protein as
sequences. It does not require a three-dimensional protein structure as
an input. Instead, it learns patterns from the encoded drug SMILES
string and protein amino-acid sequence.

The network has two separate feature-extraction branches:

-   A drug branch learns patterns from the SMILES character sequence.
-   A protein branch learns patterns from the amino-acid sequence.

The learned representations are joined and passed through fully
connected layers to produce one numerical prediction.

### 3.2 Input representations

**Drug input --- SMILES:** SMILES (Simplified Molecular Input Line Entry
System) is a text representation of a molecule. It uses characters and
symbols to describe atoms, bonds, branches and ring structures. It is
compact and convenient for computational processing. It is not a
natural-language description and should be validated before being passed
to the model.

**Protein input --- amino-acid sequence:** The protein is represented by
its amino-acid one-letter codes. This sequence is processed as an
ordered string.

The paper's preprocessing maps characters to integer identifiers and
uses padding or truncation to create fixed-length inputs. The reported
maximum sequence lengths differ by dataset:

  Dataset     Maximum SMILES length   Maximum protein sequence length
  --------- ----------------------- ---------------------------------
  Davis                          85                             1,200
  KIBA                          100                             1,000

The exact character dictionary, padding convention, truncation rule and
preprocessing order must match the model checkpoint used at inference
time. Recreating only the network layers is not sufficient if the input
encoding differs.

### 3.3 CNN feature extraction

A convolutional neural network (CNN) applies learnable filters across a
sequence to identify local patterns. In DeepDTA, separate
one-dimensional convolutional branches process the drug and protein
sequences.

The paper describes three convolutional layers in each branch, with 32,
64 and 96 filters respectively, followed by max-pooling operations. An
embedding dimension of 128 is used for the sequence representation.

After feature extraction, the drug and protein representations are
combined. Fully connected layers then learn how features from the two
inputs relate to the affinity target. The described dense-layer sizes
are 1,024, 1,024 and 512 units, with dropout of 0.1, followed by a
single output.

### 3.4 Training setup described in the paper

The paper reports the following principal training choices:

  Component       Reported setting
  --------------- --------------------------
  Activation      ReLU
  Optimizer       Adam
  Learning rate   0.001
  Loss            Mean Squared Error (MSE)
  Epochs          100
  Batch size      256
  Dropout         0.1
  Output          One continuous score

These settings describe the research implementation. They are a starting
reference, not a guarantee that the same result will be obtained in a
new environment or on a differently prepared dataset.

### 3.5 Datasets described in the paper

The paper evaluates DeepDTA using two datasets:

  Dataset     Proteins   Compounds   Drug--target measurements
  --------- ---------- ----------- ---------------------------
  Davis            442          68                      30,056
  KIBA             229       2,111                     118,254

The datasets have different measurement conventions and target values. A
prototype should keep their preprocessing, model checkpoints and output
interpretation separate.

### 3.6 Evaluation metrics

**Mean Squared Error (MSE):** Measures the average squared difference
between predicted and observed values. Lower MSE indicates smaller
squared prediction errors on the evaluation data.

\[ MSE =
`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}(y_i-`\hat{y}`{=tex}\_i)\^2
\]

Here, (y_i) is the observed value, (`\hat{y}`{=tex}\_i) is the predicted
value and (N) is the number of evaluated examples.

**Concordance Index (CI):** Measures how consistently a model ranks
pairs of examples in the same order as their observed values. A higher
CI indicates better ranking agreement. CI is a ranking metric and does
not by itself show the size of numerical prediction errors.

The paper reports the following results for its experimental setup:

  Dataset        CI     MSE
  --------- ------- -------
  Davis       0.878   0.261
  KIBA        0.863   0.194

These are the results reported by the original paper. They must not be
presented as results achieved by this project unless our own
trained/evaluated model reproduces them under a comparable data split
and evaluation protocol.

## 4. Technical Terms Used in This Project

  -----------------------------------------------------------------------
  Term                                Explanation in the context of
                                      DeepDTA
  ----------------------------------- -----------------------------------
  Deep learning                       A machine-learning approach that
                                      uses layered neural networks to
                                      learn patterns from data.

  Neural network                      A trainable mathematical model made
                                      of connected computational units
                                      and parameters.

  CNN                                 A neural network that applies
                                      filters to learn local patterns in
                                      data such as sequences or images.

  1D convolution                      A convolution operation that slides
                                      filters along one sequence
                                      dimension.

  Embedding                           A trainable vector representation
                                      assigned to an input token or
                                      character.

  Token / character encoding          Mapping input symbols, such as
                                      SMILES or amino-acid characters, to
                                      numeric identifiers that a model
                                      can process.

  Padding                             Adding designated values to shorter
                                      sequences so inputs have a common
                                      length.

  Truncation                          Cutting sequences that exceed the
                                      configured maximum length.

  Max pooling                         Keeping the maximum value from
                                      local regions, reducing the length
                                      of a feature sequence.

  Dense / fully connected layer       A layer in which each output unit
                                      receives input from all units in
                                      the preceding layer.

  ReLU                                An activation function,
                                      (f(x)=`\max`{=tex}(0,x)), that
                                      introduces non-linearity.

  Dropout                             A training technique that
                                      temporarily disables a fraction of
                                      units to help reduce overfitting.

  Epoch                               One complete pass through the
                                      training dataset.

  Batch size                          Number of training examples
                                      processed together before an
                                      optimisation update.

  Optimizer                           The algorithm used to update model
                                      parameters during training.

  Adam                                An adaptive gradient-based
                                      optimisation algorithm.

  Loss function                       A value measuring how far
                                      predictions are from the target
                                      values during training.

  Regression                          Predicting a numerical value rather
                                      than a class label.

  Overfitting                         When a model learns
                                      training-specific patterns and
                                      performs poorly on unseen data.

  Inference                           Using a trained model to make a
                                      prediction on new input data.

  Checkpoint                          A saved model state or model file
                                      used later for inference or
                                      continued training.

  API                                 A defined interface through which
                                      software components exchange
                                      requests and responses.

  Backend                             The server-side part of the
                                      application that validates inputs,
                                      runs the model and returns results.

  Frontend                            The user-facing part of the
                                      application, such as the website
                                      interface.

  FastAPI                             A Python framework for creating web
                                      APIs.

  React                               A JavaScript library for building
                                      user interfaces.

  TensorFlow / Keras                  Tools for building, training,
                                      saving and running neural-network
                                      models.

  RDKit                               A cheminformatics toolkit that can
                                      parse and inspect molecular
                                      structures and SMILES.
  -----------------------------------------------------------------------

## 5. Our Proposed Project Approach

This section describes the implementation plan for our prototype. It
distinguishes our planned application architecture from the original
research paper's model design.

### 5.1 Overall workflow

1.  The user opens the DeepDTA website.
2.  The user enters a drug SMILES string.
3.  The user provides a protein amino-acid sequence or, if implemented,
    a supported protein identifier that the backend can resolve to a
    sequence.
4.  The frontend sends the inputs and selected model/dataset to the
    backend API.
5.  The backend validates the request and applies the exact
    preprocessing expected by the selected model.
6.  The trained model performs inference.
7.  The backend returns the predicted score and relevant metadata.
8.  The frontend displays the result, selected dataset, score
    interpretation and limitations.

### 5.2 Proposed technology stack

  -----------------------------------------------------------------------
  Layer                   Proposed technology     Responsibility
  ----------------------- ----------------------- -----------------------
  Frontend                React, HTML, CSS,       Forms, navigation,
                          JavaScript              validation feedback and
                                                  result display

  UI styling              CSS or Tailwind CSS     Responsive layout and
                                                  visual design

  Backend API             Python and FastAPI      Request handling,
                                                  validation and
                                                  prediction endpoints

  Model runtime           TensorFlow / Keras      Loading the trained
                                                  model and performing
                                                  inference

  Data processing         Python, NumPy, Pandas   Preparing and handling
                                                  input data

  Chemistry validation    RDKit                   Checking whether a
  (optional)                                      submitted SMILES string
                                                  can be parsed

  Storage (optional)      SQLite                  Saving prediction
                                                  history and basic
                                                  metadata

  Charts (optional)       Plotly                  Visualising evaluation
                                                  metrics and project
                                                  results
  -----------------------------------------------------------------------

### 5.3 Model and preprocessing strategy

The prototype should use a trained DeepDTA-compatible model and its
matching preprocessing configuration. For each supported dataset, keep
the following together:

-   The model checkpoint.
-   The character/token dictionaries.
-   Maximum input lengths.
-   Padding and truncation rules.
-   Dataset-specific output interpretation.
-   Model version and training/evaluation information.

If the original research implementation and compatible model files are
used, document their source and licence. If the model is retrained,
document the dataset version, data split, hyperparameters, random seeds
and evaluation results.

**Do not assume that a newly written model with similar layer sizes is
identical to the original DeepDTA model.** Input dictionaries, layer
order, pooling, tensor shapes, weight files and preprocessing must all
be compatible.

### 5.4 Dataset-specific output handling

The application should keep Davis and KIBA predictions separate.

-   **Davis model:** display the predicted pKd value if the model has
    been trained to predict pKd. A Kd conversion may be shown only when
    the units and transformation are confirmed. For pKd,
    (Kd(M)=10\^{-pKd}).
-   **KIBA model:** display the predicted KIBA score. Do not label it
    pKd, Kd, Ki or IC50.
-   Clearly show which model was used and avoid comparing scores from
    different scales as if they were equivalent.

### 5.5 Backend API design

A simple initial API design could include:

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  GET                     `/`                     Confirm that the
                                                  backend is running

  GET                     `/health`               Report API/model
                                                  readiness

  POST                    `/predict`              Validate inputs and
                                                  return a prediction

  GET                     `/models`               List the prediction
                                                  models supported by the
                                                  application
  -----------------------------------------------------------------------

An example request body for a prediction could be:

``` json
{
  "smiles": "CCO",
  "protein_sequence": "MKWVTF...",
  "dataset": "davis"
}
```

This is an illustrative API shape. The final request schema should match
the actual preprocessing and model interface.

An example response could be:

``` json
{
  "dataset": "davis",
  "predicted_score": 7.82,
  "score_type": "pKd",
  "model_version": "your-model-version"
}
```

The example score above is illustrative only; it is not a real
prediction or a validated project result.

### 5.6 Input validation and error handling

The backend should check that:

-   The SMILES field is present and not empty.
-   The protein sequence is present and not empty.
-   The selected dataset/model is supported.
-   The protein sequence contains only accepted amino-acid symbols
    according to the model's preprocessing rules.
-   Input lengths are handled using the exact model-specific
    padding/truncation rules.
-   SMILES parsing succeeds if RDKit validation is enabled.
-   The model checkpoint and its preprocessing files are available.

Errors should be returned as clear messages. Invalid inputs should not
be silently converted into apparently valid predictions.

## 6. Proposed Website Design

The website should look like a scientific research application: clear,
restrained and easy to use. A dark navy and blue palette is one possible
design choice, but readability and accessibility are more important than
decoration.

### 6.1 Main pages

**1. Home page** - Project name and a short explanation of DeepDTA. - A
simple diagram showing drug input + protein input → model → predicted
affinity score. - A clear link to the prediction page. - A note that
predictions require experimental validation.

**2. Prediction page** - SMILES input field. - Protein sequence input
field. - Dataset/model selector (Davis or KIBA, only when both are
actually available). - Input guidance and validation messages. - Predict
button and loading state.

**3. Results page or result panel** - Predicted score. - Score type and
dataset name. - Model version. - A short explanation of the score's
meaning. - Input summary and any validation warnings. - Clear disclaimer
that the output is a model estimate.

**4. How It Works page** - Biological background. - Explanation of
SMILES and protein sequences. - Dual-branch CNN architecture. -
Preprocessing and prediction workflow.

**5. Research / About page** - Research-paper citation. - Dataset
information and data provenance. - Model architecture and implementation
details. - Evaluation protocol and measured project results. -
Limitations and acknowledgements.

**6. Analytics page (optional)** - Project's own evaluation metrics. -
Actual training/validation curves, if generated. - Dataset summaries. -
Do not display the paper's reported scores as if they were our own.

### 6.2 Suggested user journey

A first-time user should be able to understand the purpose of the tool,
enter a drug and protein, choose a supported model, receive a prediction
and understand what the score does and does not mean. Avoid overwhelming
the user with raw technical details on the prediction screen; put the
deeper explanation on the How It Works and Research pages.

## 7. Suggested Repository Structure

One possible structure is:

``` text
DeepDTA/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── project_report.md
│
├── platform/
│   ├── backend/
│   │   ├── main.py
│   │   ├── predict.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   ├── requirements.txt
│   │   └── models/
│   │       ├── davis/
│   │       └── kiba/
│   │
│   └── frontend/
│       ├── package.json
│       ├── public/
│       └── src/
│           ├── components/
│           ├── pages/
│           ├── services/
│           └── styles/
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── config/
│
├── datasets/
│   └── README.md
│
└── notebooks/
```

This is a suggested structure, not a statement that every file or module
has already been implemented. Large datasets and model checkpoints may
need to be stored outside GitHub if they exceed repository limits or
have redistribution restrictions. Never commit API keys, passwords,
private data or local virtual-environment folders.

## 8. Development Roadmap

  -----------------------------------------------------------------------
  Phase                   Work to complete        Expected outcome
  ----------------------- ----------------------- -----------------------
  1\. Environment setup   Configure a compatible  Reproducible
                          Python environment and  development environment
                          install project         
                          dependencies            

  2\. Research/code       Study the original      Clear understanding of
  inspection              implementation,         model inputs and
                          datasets, preprocessing outputs
                          and model files         

  3\. Model preparation   Obtain or train         Model ready for
                          compatible Davis/KIBA   inference
                          model checkpoints       

  4\. Prediction testing  Test known examples and Reliable prediction
                          verify preprocessing,   function
                          output shape and errors 

  5\. Backend development Build FastAPI endpoints Working prediction API
                          and validation          

  6\. Frontend            Build React pages and   Usable website
  development             input forms             interface

  7\. Integration         Connect React to        End-to-end prototype
                          FastAPI and display     
                          real model responses    

  8\. Evaluation          Evaluate on a defined   Honest performance
                          held-out set and report report
                          CI/MSE                  

  9\. Documentation and   Document setup, usage,  Shareable GitHub
  deployment              limitations and         project
                          deployment steps        
  -----------------------------------------------------------------------

## 9. Testing and Evaluation Plan

Testing should cover both software correctness and model performance.

### 9.1 Functional tests

-   Confirm the API starts and its health endpoint responds.
-   Submit a valid SMILES string and protein sequence.
-   Submit empty inputs and check that clear validation errors appear.
-   Submit malformed SMILES and check the response if chemical
    validation is enabled.
-   Submit unsupported dataset/model names.
-   Test sequences shorter and longer than the configured model length.
-   Confirm that Davis and KIBA select the correct model and
    preprocessing.
-   Confirm that the frontend displays the returned score type and
    dataset correctly.

### 9.2 Model evaluation

Evaluate the model on a held-out test set that was not used for fitting
or model selection. Record the data split and evaluation procedure.
Report at least MSE and CI if they are part of the selected evaluation
protocol. Where possible, compare results with the original paper only
when the dataset processing and evaluation setup are sufficiently
comparable.

Keep a record of: - Dataset and version. - Training, validation and test
split. - Model architecture and checkpoint version. - Preprocessing
configuration. - Hyperparameters and random seeds. - Evaluation metrics
and date. - Known limitations.

## 10. Limitations and Responsible Use

1.  **Prediction is not experimental confirmation.** A predicted score
    must be validated experimentally.
2.  **Binding affinity is not clinical effectiveness.** A strong
    predicted interaction does not establish safety, selectivity,
    dosage, efficacy or patient benefit.
3.  **Sequence-based modelling has limitations.** The original DeepDTA
    approach learns from sequence representations and does not
    explicitly model the complete three-dimensional structure, binding
    pose or molecular dynamics.
4.  **Dataset coverage matters.** A model may perform poorly on
    proteins, compounds or chemical patterns that differ from its
    training data.
5.  **Assay values differ.** Kd, Ki, IC50 and KIBA scores have different
    definitions and must not be treated as interchangeable.
6.  **Preprocessing compatibility is essential.** Incorrect token
    dictionaries, padding, truncation or input ordering can invalidate
    predictions.
7.  **Model outputs have uncertainty.** A single score should not be
    presented as certainty. The prototype should communicate that
    predictions are estimates.
8.  **Data and model licences must be respected.** Check the terms of
    use before redistributing datasets, checkpoints or third-party code.

## 11. Expected Outcome

The intended outcome is a web-based prototype that demonstrates the
DeepDTA drug--target binding-affinity prediction workflow. It will
accept a drug SMILES representation and a target protein sequence, pass
them through a compatible trained model, and display the resulting
dataset-specific prediction.

The prototype will also explain the biological background, the model
architecture, the meaning of the output and the limitations of
computational predictions. Its usefulness will be assessed through
successful end-to-end operation, correct input handling and transparent
model evaluation.

The expected outcome is a research and educational prototype, not a
clinically validated drug-discovery system.

## 12. References

1.  Öztürk, H., Özgür, A. and Ozkirimli, E. (2018). **DeepDTA: Deep
    Drug--Target Binding Affinity Prediction.** *Bioinformatics*.\
2.  Original research implementation:
    https://github.com/hkmztrk/DeepDTA\
3.  UniProt --- protein sequence and functional information:
    https://www.uniprot.org/\
4.  PubChem --- chemical compound information:
    https://pubchem.ncbi.nlm.nih.gov/\
5.  RDKit --- cheminformatics toolkit documentation:
    https://www.rdkit.org/docs/\
6.  TensorFlow documentation: https://www.tensorflow.org/\
7.  FastAPI documentation: https://fastapi.tiangolo.com/\
8.  React documentation: https://react.dev/

------------------------------------------------------------------------


