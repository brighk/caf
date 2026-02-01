"""
Prometheus metrics for CAF system monitoring.
Tracks verification latency, refinement iterations, and system health.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any


# Request metrics
inference_requests_total = Counter(
    'caf_inference_requests_total',
    'Total number of inference requests',
    ['status', 'endpoint']
)

verification_attempts_total = Counter(
    'caf_verification_attempts_total',
    'Total number of verification attempts',
    ['result']
)

refinement_iterations = Histogram(
    'caf_refinement_iterations',
    'Number of refinement iterations per request',
    buckets=[0, 1, 2, 3, 5, 10]
)

# Latency metrics
inference_latency_seconds = Histogram(
    'caf_inference_latency_seconds',
    'Inference engine latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

parsing_latency_seconds = Histogram(
    'caf_parsing_latency_seconds',
    'Semantic parsing latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

verification_latency_seconds = Histogram(
    'caf_verification_latency_seconds',
    'Truth anchor verification latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

validation_latency_seconds = Histogram(
    'caf_validation_latency_seconds',
    'Causal validation latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

total_pipeline_latency_seconds = Histogram(
    'caf_total_pipeline_latency_seconds',
    'Total pipeline latency in seconds',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# Quality metrics
triplet_extraction_count = Histogram(
    'caf_triplet_extraction_count',
    'Number of triplets extracted per request',
    buckets=[0, 1, 3, 5, 10, 20, 50]
)

verification_similarity_score = Histogram(
    'caf_verification_similarity_score',
    'Similarity score from verification',
    buckets=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

causal_violations = Counter(
    'caf_causal_violations_total',
    'Total number of causal violations detected',
    ['violation_type']
)

# System health metrics
component_health = Gauge(
    'caf_component_health',
    'Component health status (1=healthy, 0=unhealthy)',
    ['component']
)

gpu_memory_usage_bytes = Gauge(
    'caf_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

model_load_time_seconds = Gauge(
    'caf_model_load_time_seconds',
    'Time taken to load the LLM model',
    ['model_name']
)

# Knowledge base metrics
kb_triplet_count = Gauge(
    'caf_kb_triplet_count',
    'Total number of triplets in knowledge base'
)

vector_db_size = Gauge(
    'caf_vector_db_size',
    'Number of entities in vector database'
)

# System info
system_info = Info(
    'caf_system',
    'CAF system information'
)


class MetricsCollector:
    """Helper class for collecting and recording metrics"""

    @staticmethod
    def record_inference_request(status: str, endpoint: str):
        """Record an inference request"""
        inference_requests_total.labels(status=status, endpoint=endpoint).inc()

    @staticmethod
    def record_verification(result: str):
        """Record a verification attempt"""
        verification_attempts_total.labels(result=result).inc()

    @staticmethod
    def record_refinement_count(count: int):
        """Record number of refinement iterations"""
        refinement_iterations.observe(count)

    @staticmethod
    def record_latencies(metrics: Dict[str, float]):
        """Record all latency metrics"""
        if 'inference' in metrics:
            inference_latency_seconds.observe(metrics['inference'])

        if 'parsing' in metrics:
            parsing_latency_seconds.observe(metrics['parsing'])

        if 'verification' in metrics:
            verification_latency_seconds.observe(metrics['verification'])

        if 'validation' in metrics:
            validation_latency_seconds.observe(metrics['validation'])

        if 'total' in metrics:
            total_pipeline_latency_seconds.observe(metrics['total'])

    @staticmethod
    def record_triplet_count(count: int):
        """Record number of extracted triplets"""
        triplet_extraction_count.observe(count)

    @staticmethod
    def record_similarity(score: float):
        """Record verification similarity score"""
        verification_similarity_score.observe(score)

    @staticmethod
    def record_causal_violation(violation_type: str):
        """Record a causal violation"""
        causal_violations.labels(violation_type=violation_type).inc()

    @staticmethod
    def update_component_health(component: str, is_healthy: bool):
        """Update component health status"""
        component_health.labels(component=component).set(1 if is_healthy else 0)

    @staticmethod
    def update_gpu_memory(gpu_id: str, bytes_used: int):
        """Update GPU memory usage"""
        gpu_memory_usage_bytes.labels(gpu_id=gpu_id).set(bytes_used)

    @staticmethod
    def set_model_load_time(model_name: str, seconds: float):
        """Set model load time"""
        model_load_time_seconds.labels(model_name=model_name).set(seconds)

    @staticmethod
    def update_kb_stats(triplet_count: int, vector_db_count: int):
        """Update knowledge base statistics"""
        kb_triplet_count.set(triplet_count)
        vector_db_size.set(vector_db_count)

    @staticmethod
    def set_system_info(info: Dict[str, str]):
        """Set system information"""
        system_info.info(info)
