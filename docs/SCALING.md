# Guía de Escalabilidad - Sistema Biométrico

## Capacidad por Fase

| Fase | Arquitectura | Huellas | Búsquedas/seg | Latencia P99 |
|------|-------------|---------|---------------|--------------|
| MVP | pgvector single-node | 100K-1M | 50-100 | <100ms |
| Escala Media | pgvector + sharding | 1M-10M | 500+ | <150ms |
| Escala Grande | pgvector + Citus | 10M-100M | 2K+ | <200ms |
| Escala Masiva | Milvus/Qdrant | 100M-1B+ | 10K+ | <50ms |

## Fase 1: MVP Single-Node (Actual)

### Configuración
```sql
-- Extensión pgvector
CREATE EXTENSION vector;

-- Índice IVFFlat
CREATE INDEX ON fingerprint_vectors 
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);  -- Ajustar según tamaño
```

### Tuning de Listas IVFFlat
```python
# Regla general: lists = sqrt(total_rows)
lists = int(math.sqrt(total_vectors))

# Ejemplos:
# 10K vectores → lists = 100
# 100K vectores → lists = 316
# 1M vectores → lists = 1000
```

### Límites
- **Capacidad**: ~1-5M vectores
- **RAM**: 16-32GB recomendado
- **Storage**: SSD obligatorio
- **CPU**: 4-8 cores

## Fase 2: Sharding Horizontal

### Estrategias de Sharding

#### Opción A: Por Región Geográfica
```python
# Router personalizado
def get_shard(person_id: str) -> str:
    region = person_id[:2]  # PE, CO, CL, etc.
    shard_map = {
        "PE": "shard_peru",
        "CO": "shard_colombia",
        "CL": "shard_chile",
    }
    return shard_map.get(region, "shard_default")
```

#### Opción B: Por Hash de ID
```python
def get_shard(person_id: str) -> int:
    hash_value = hashlib.md5(person_id.encode()).hexdigest()
    shard_id = int(hash_value, 16) % NUM_SHARDS
    return shard_id
```

#### Opción C: Por Rango de Documento
```python
def get_shard(document: str) -> int:
    doc_num = int(document)
    if doc_num < 50_000_000:
        return 0
    elif doc_num < 100_000_000:
        return 1
    else:
        return 2
```

### Implementación

```python
# src/storage/sharded_repository.py
class ShardedFingerprintRepository:
    def __init__(self, shards: List[FingerprintRepository]):
        self.shards = shards
    
    def register(self, fingerprint, person_id, name, document):
        shard_idx = self._get_shard(person_id)
        return self.shards[shard_idx].register(
            fingerprint, person_id, name, document
        )
    
    def identify(self, fingerprint):
        # Buscar en todos los shards en paralelo
        with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
            futures = [
                executor.submit(shard.identify, fingerprint)
                for shard in self.shards
            ]
            results = [f.result() for f in futures]
        
        # Retornar el mejor match
        return min(results, key=lambda r: r.distance)
```

### Configuración Multi-DB

```yaml
# docker-compose.sharded.yml
services:
  postgres_shard_1:
    image: postgres:15
    environment:
      POSTGRES_DB: fingerprint_shard_1
    volumes:
      - pg_shard_1:/var/lib/postgresql/data
  
  postgres_shard_2:
    image: postgres:15
    environment:
      POSTGRES_DB: fingerprint_shard_2
    volumes:
      - pg_shard_2:/var/lib/postgresql/data
  
  api:
    environment:
      SHARD_1_URL: postgresql://postgres@postgres_shard_1/fingerprint_shard_1
      SHARD_2_URL: postgresql://postgres@postgres_shard_2/fingerprint_shard_2
```

## Fase 3: Citus (Sharding Nativo)

### Setup
```bash
# Añadir a requirements.txt
citus>=12.0
```

```sql
-- Habilitar Citus
CREATE EXTENSION citus;

-- Convertir tabla en distribuida
SELECT create_distributed_table(
    'fingerprint_vectors', 
    'person_id',  -- Clave de distribución
    colocate_with => 'fingerprints'
);

-- Citus maneja sharding automáticamente
```

### Ventajas
- ✅ Sharding transparente
- ✅ Joins distribuidos
- ✅ Balanceo automático
- ✅ Queries SQL estándar

## Fase 4: Migración a Milvus

### Cuándo migrar
- >50M vectores
- Latencia P99 >200ms con pgvector
- Necesitas GPU acceleration
- Búsquedas >10K/segundo

### Plan de Migración

```python
# 1. Exportar vectores de PostgreSQL
def export_vectors():
    session = db_manager.get_session()
    vectors = session.query(FingerprintVector).all()
    
    return [
        {
            "id": v.id,
            "vector": v.embedding,
            "metadata": get_metadata(v.id)
        }
        for v in vectors
    ]

# 2. Importar a Milvus
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")

collection = Collection("fingerprints")
collection.insert([
    vec["id"] for vec in vectors,
    vec["vector"] for vec in vectors
])

collection.create_index(
    field_name="vector",
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048}
    }
)
```

### Arquitectura con Milvus

```
┌─────────────┐
│  FastAPI    │
└──────┬──────┘
       │
   ┌───┴────┐
   │        │
┌──▼──┐  ┌─▼──────┐
│Milvus│  │Postgres│
│Vector│  │Metadata│
│Search│  │ Only   │
└──────┘  └────────┘
```

## Monitoreo y Métricas

### KPIs Críticos

```python
# Métricas a monitorear
metrics_to_track = {
    "vector_count": "Total de vectores indexados",
    "search_latency_p50": "Latencia P50 de búsqueda",
    "search_latency_p99": "Latencia P99 de búsqueda",
    "throughput": "Búsquedas por segundo",
    "index_size_mb": "Tamaño del índice en MB",
    "db_connections": "Conexiones activas a DB",
}
```

### Alertas Recomendadas

```yaml
alerts:
  - name: high_latency
    condition: search_latency_p99 > 500ms
    action: Considerar sharding o migración
  
  - name: near_capacity
    condition: vector_count > 5M (single-node)
    action: Planear sharding
  
  - name: connection_saturation
    condition: db_connections > 80% pool_size
    action: Aumentar pool o escalar
```

## Costos Estimados

### Single-Node (Fase 1)
- **Infraestructura**: $50-200/mes
- **PostgreSQL**: 16GB RAM, 4 cores, 500GB SSD
- **Capacidad**: 1-5M vectores

### Sharding (Fase 2)
- **Infraestructura**: $200-800/mes
- **3-5 shards**: 8GB RAM cada uno
- **Capacidad**: 10-50M vectores

### Milvus (Fase 3)
- **Infraestructura**: $1000-5000/mes
- **Cluster**: 3+ nodos, GPUs opcionales
- **Capacidad**: 100M-1B+ vectores

## Checklist de Migración

### Antes de Escalar
- [ ] Monitoreo de métricas activo
- [ ] Backups automáticos configurados
- [ ] Tests de carga ejecutados
- [ ] Plan de rollback documentado
- [ ] Equipo entrenado en nueva arquitectura

### Durante Migración
- [ ] Migración en modo dual-write (ambos sistemas)
- [ ] Validación de consistencia
- [ ] Tests A/B con tráfico real
- [ ] Monitoreo intensivo 24/7

### Después de Migración
- [ ] Desmantelar sistema antiguo después de 2 semanas
- [ ] Documentar lecciones aprendidas
- [ ] Actualizar runbooks operacionales
- [ ] Optimizar configuración basada en métricas reales

## Recomendaciones Finales

1. **No optimizar prematuramente**: Empezar con pgvector single-node
2. **Medir constantemente**: Métricas son críticas para decidir cuándo escalar
3. **Planear con anticipación**: Migración toma 2-3 meses
4. **Validar en producción**: Tests de carga no replican comportamiento real
5. **Mantener simplicidad**: Solo añadir complejidad cuando sea absolutamente necesario
