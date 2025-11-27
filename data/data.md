# Data Overview

## Datasets Used
This project uses two open biomedical ontologies in **OWL** format:

1. **Human Phenotype Ontology (HPO)**
   - **Description:** A structured vocabulary describing human phenotypic abnormalities (observable traits and clinical features).  
   - **Classes:** ~31,860  
   - **Roots:** 522  
   - **Max Depth:** 17  

2. **Mammalian Phenotype Ontology (MP)**
   - **Description:** A complementary ontology capturing phenotypic information in mouse and other mammalian models.  
   - **Classes:** ~36,182  
   - **Roots:** 540  
   - **Max Depth:** 34  

## How to Obtain the Data
Both ontologies are publicly available under open licenses via the **OBO Foundry**.

You can download them directly:
```bash
curl -L http://purl.obolibrary.org/obo/hp.owl -o hp.owl
curl -L http://purl.obolibrary.org/obo/mp.owl -o mp.owl
