# Context-Aware-Transformer
## Abstract
We propose a multi-horizon forecasting approach that accurately models the underlying patterns on different time scales. Our approach is based on the transformer architecture, which across a wide range of domains, has demonstrated significant improvements over other architectures.
Several approaches focus on integrating a temporal context into the query-key similarity of the attention mechanism of transformers to further improve their forecasting quality. In this paper, we provide several extensions to this line of work. 
We propose an adjustable context-aware attention that dynamically learns the ideal temporal context length for each forecasting time point. This allows the model to seamlessly switch between different time scales as needed, hence providing users with a better forecasting model. 
Furthermore, we exploit redundancies arising from incorporating the temporal context into the attention mechanism to improve runtime and space complexity. Our experiments on several real-world datasets demonstrate significant performance improvements over existing state-of-the-art methodologies. 
