"""RAG engine for retrieving clinical guidance from Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.core.config import Settings
from src.schemas.vitals import PatientVitals


@dataclass(frozen=True)
class RetrievalRoute:
    """Retrieval routing profile for a likely ICU syndrome."""

    name: str
    summary: str
    query_focus: str
    preferred_conditions: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalChunk:
    """Debug-friendly view of a retrieved guideline chunk."""

    rank: int
    score: int
    condition: str
    source: str
    chunk_index: str
    preview: str


class RAGEngine:
    """Simple RAG engine wrapper for clinical guidance retrieval."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the RAG engine.

        Args:
            settings: Loaded application settings.
        """

        self._settings: Settings = settings
        self._logger = logging.getLogger("cloud_icu_sentinel.rag_engine")
        self._embeddings: OpenAIEmbeddings | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._llm: ChatOpenAI | None = None

    def retrieve_guideline_context(self, vitals: PatientVitals) -> str:
        """Retrieve top-k semantic context from Qdrant or fall back to local markdown.

        Args:
            vitals: The patient vitals under analysis.

        Returns:
            str: Retrieved or fallback guideline context.
        """

        return self.inspect_retrieval(vitals)["guideline_context"]

    def inspect_retrieval(self, vitals: PatientVitals) -> dict[str, object]:
        """Return route, query, retrieved chunks, and formatted context for debugging."""

        route = self._build_retrieval_route(vitals)
        query = self._build_retrieval_query(vitals, route)

        if self._settings.openai_api_key is None:
            return self._build_fallback_inspection(
                vitals=vitals,
                route=route,
                query=query,
                reason="openai_api_key_missing",
            )

        try:
            scored_documents = self._retrieve_scored_documents(vitals, route)
            documents = [document for document, _score in scored_documents]
            if not documents:
                return self._build_fallback_inspection(
                    vitals=vitals,
                    route=route,
                    query=query,
                    reason="no_qdrant_results",
                )

            return {
                "mode": "qdrant",
                "route_name": route.name,
                "route_summary": route.summary,
                "query": query,
                "chunks": self._build_chunk_debug_payload(scored_documents),
                "guideline_context": self._format_documents(documents),
            }
        except Exception as exc:
            self._logger.warning("RAG retrieval failed, falling back to local knowledge base: %s", exc)
            return self._build_fallback_inspection(
                vitals=vitals,
                route=route,
                query=query,
                reason=f"qdrant_error: {exc}",
            )

    def build_recommendations(
        self,
        vitals: PatientVitals,
        guideline_context: str | None = None,
    ) -> List[str]:
        """Generate LLM-backed intervention suggestions with guarded fallbacks."""

        if self._settings.openai_api_key is None:
            return self._build_fallback_recommendations(vitals)

        context = guideline_context or self.retrieve_guideline_context(vitals)

        try:
            chain = self._build_recommendation_chain()
            response = chain.invoke(
                {
                    "vitals_snapshot": self._build_vitals_snapshot(vitals),
                    "status": vitals.status,
                    "context": context,
                }
            )
        except Exception as exc:
            self._logger.warning(
                "LLM recommendation generation failed for patient status '%s': %s",
                vitals.status,
                exc,
            )
            return self._build_fallback_recommendations(vitals)

        recommendations = self._normalize_recommendations(response.content)
        if recommendations:
            return recommendations

        self._logger.warning(
            "LLM recommendation generation returned no usable content for patient status '%s'",
            vitals.status,
        )
        return self._build_fallback_recommendations(vitals)

    def _retrieve_documents(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> List[Document]:
        """Run a semantic similarity search against Qdrant."""

        return [
            document
            for document, _score in self._retrieve_scored_documents(vitals, route)
        ]

    def _retrieve_scored_documents(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> List[tuple[Document, int]]:
        """Run retrieval and keep route-aware scores for debugging and re-ranking."""

        vector_store = self._build_vector_store()
        query = self._build_retrieval_query(vitals, route)
        candidate_k = min(max(self._settings.qdrant_top_k * 2, self._settings.qdrant_top_k), 10)
        documents = vector_store.similarity_search(
            query=query,
            k=candidate_k,
        )
        return self._rerank_documents(documents, route)

    def _build_retrieval_query(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> str:
        """Build a retrieval query from the current patient snapshot."""

        return (
            f"ICU patient clinical guidance for {route.summary}. "
            f"Patient status: {vitals.status}. "
            f"{self._build_vitals_snapshot(vitals)}. "
            f"Focus on {route.query_focus}. "
            "Prioritize ICU bedside stabilization, contraindications, and escalation guidance."
        )

    def _build_retrieval_route(self, vitals: PatientVitals) -> RetrievalRoute:
        """Infer the most likely retrieval focus from the current patient snapshot."""

        status_text = vitals.status.lower()

        routes = (
            RetrievalRoute(
                name="severe_hypoglycemia",
                summary="possible severe hypoglycemia with neuroglycopenia",
                query_focus="rapid glucose correction, airway protection, bedside glucose checks, and contraindications in severe hypoglycemia",
                preferred_conditions=("severe hypoglycemia",),
                keywords=(
                    "hypoglycemia",
                    "low glucose",
                    "neuroglycopenia",
                    "confusion",
                    "diaphoresis",
                    "insulin",
                    "seizure",
                ),
            ),
            RetrievalRoute(
                name="early_septic_shock",
                summary="possible early septic shock with ICU deterioration",
                query_focus="source evaluation, cultures, broad sepsis bundle priorities, fluids, vasopressors, and senior escalation",
                preferred_conditions=("early septic shock",),
                keywords=(
                    "sepsis",
                    "septic",
                    "infection",
                    "shock",
                    "fever",
                    "lactate",
                    "vasopressor",
                    "hypotension",
                ),
            ),
            RetrievalRoute(
                name="psvt",
                summary="possible paroxysmal supraventricular tachycardia",
                query_focus="regular narrow-complex tachycardia assessment, vagal maneuvers, adenosine precautions, and electrical cardioversion escalation",
                preferred_conditions=("psvt",),
                keywords=(
                    "psvt",
                    "svt",
                    "supraventricular tachycardia",
                    "palpitations",
                    "narrow complex",
                    "regular tachycardia",
                ),
            ),
            RetrievalRoute(
                name="acute_left_heart_failure_pulmonary_edema",
                summary="possible acute left heart failure with pulmonary edema",
                query_focus="pulmonary edema stabilization, noninvasive ventilation, diuresis precautions, nitrates cautions, and cardiology escalation",
                preferred_conditions=("acute left heart failure pulmonary edema",),
                keywords=(
                    "pulmonary edema",
                    "flash edema",
                    "orthopnea",
                    "frothy sputum",
                    "heart failure",
                    "cardiogenic",
                ),
            ),
            RetrievalRoute(
                name="hypovolemic_shock_postoperative_hemorrhage",
                summary="possible postoperative hemorrhage with hypovolemic shock",
                query_focus="bleeding source control, transfusion priorities, volume resuscitation, and urgent surgical escalation",
                preferred_conditions=("hypovolemic shock postoperative hemorrhage",),
                keywords=(
                    "hemorrhage",
                    "bleeding",
                    "postoperative",
                    "post-op",
                    "hypovolemia",
                    "blood loss",
                    "shock",
                ),
            ),
        )

        route_match_scores = [
            (sum(1 for keyword in route.keywords if keyword in status_text), route)
            for route in routes
        ]
        best_match_score, best_match_route = max(route_match_scores, key=lambda item: item[0])
        if best_match_score > 0:
            self._logger.info(
                "Using retrieval route '%s' for patient_id=%s based on status '%s' with match score=%s",
                best_match_route.name,
                vitals.patient_id,
                vitals.status,
                best_match_score,
            )
            return best_match_route

        if vitals.heart_rate >= 150.0:
            route = next(candidate for candidate in routes if candidate.name == "psvt")
            self._logger.info(
                "Using retrieval route '%s' for patient_id=%s based on tachycardia threshold",
                route.name,
                vitals.patient_id,
            )
            return route

        if any(term in status_text for term in ("edema", "orthopnea", "heart failure", "cardiogenic")):
            route = next(
                candidate
                for candidate in routes
                if candidate.name == "acute_left_heart_failure_pulmonary_edema"
            )
            self._logger.info(
                "Using retrieval route '%s' for patient_id=%s based on cardiopulmonary terms",
                route.name,
                vitals.patient_id,
            )
            return route

        default_route = RetrievalRoute(
            name="acute_respiratory_failure",
            summary="acute hypoxemic respiratory deterioration",
            query_focus="hypoxemia, respiratory distress, acute respiratory failure, oxygen escalation, and bedside intervention priorities",
            preferred_conditions=("acute respiratory failure",),
            keywords=(
                "hypoxemia",
                "respiratory",
                "oxygen",
                "spo2",
                "failure",
                "desaturation",
                "airway",
            ),
        )
        self._logger.info(
            "Using default retrieval route '%s' for patient_id=%s",
            default_route.name,
            vitals.patient_id,
        )
        return default_route

    def _rerank_documents(
        self,
        documents: List[Document],
        route: RetrievalRoute,
    ) -> List[tuple[Document, int]]:
        """Re-rank retrieved documents to favor route-consistent guideline chunks."""

        ranked_documents = sorted(
            ((document, self._score_document(document, route)) for document in documents),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked_documents[: self._settings.qdrant_top_k]

    def _score_document(self, document: Document, route: RetrievalRoute) -> int:
        """Assign a simple route-aware ranking score to a retrieved document."""

        metadata = document.metadata
        condition = str(metadata.get("condition", "")).lower()
        source = str(metadata.get("file_name", metadata.get("source", ""))).lower()
        haystack = f"{condition} {source} {document.page_content}".lower()

        score = 0
        if condition in route.preferred_conditions:
            score += 10
        if route.name.replace("_", " ") in haystack:
            score += 5

        score += sum(1 for keyword in route.keywords if keyword in haystack)
        return score

    def _build_vector_store(self) -> QdrantVectorStore:
        """Create or reuse a Qdrant vector store for similarity retrieval."""

        if self._vector_store is not None:
            return self._vector_store

        self._vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=self._settings.qdrant_collection_name,
            embedding=self._build_embeddings(),
            url=self._settings.qdrant_url,
            api_key=self._get_qdrant_api_key(),
        )
        return self._vector_store

    def _build_embeddings(self) -> OpenAIEmbeddings:
        """Create or reuse the embedding model for retrieval."""

        if self._embeddings is not None:
            return self._embeddings

        self._embeddings = OpenAIEmbeddings(
            api_key=self._settings.openai_api_key.get_secret_value(),
            model=self._settings.openai_embedding_model,
        )
        return self._embeddings

    def _build_recommendation_chain(self):
        """Create or reuse the prompt + chat model chain for recommendations."""

        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self._settings.openai_api_key.get_secret_value(),
                model=self._settings.openai_model,
                temperature=0.0,
                timeout=self._settings.openai_timeout_seconds,
                max_retries=2,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a critical care AI assistant. You MUST strictly follow the provided clinical guidelines. "
                    "Do NOT invent medication dosages. If the guidelines do not cover the situation, explicitly state "
                    "'Requires immediate human physician judgment'. Return 3 to 5 concise bullet points only.",
                ),
                (
                    "human",
                    "Patient Vitals:\n{vitals_snapshot}\nStatus: {status}\n\nRetrieved Guidelines Context:\n{context}",
                ),
            ]
        )
        return prompt | self._llm

    def _get_qdrant_api_key(self) -> str | None:
        """Return the configured Qdrant API key if present."""

        return (
            self._settings.qdrant_api_key.get_secret_value()
            if self._settings.qdrant_api_key is not None
            else None
        )

    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved Qdrant documents into a single context string."""

        formatted_chunks: list[str] = []
        for index, document in enumerate(documents, start=1):
            if not document.page_content.strip():
                continue

            metadata = document.metadata
            condition = metadata.get("condition", "unknown condition")
            source = metadata.get("file_name", metadata.get("source", "unknown source"))
            chunk_index = metadata.get("chunk_index", "n/a")
            formatted_chunks.append(
                f"[Retrieved Guideline {index} | Condition: {condition} | Source: {source} | Chunk: {chunk_index}]\n"
                f"{document.page_content.strip()}"
            )

        return "\n\n".join(formatted_chunks)

    def _load_fallback_guideline(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> str:
        """Load locally ranked markdown guidelines as a fallback retrieval source.

        Returns:
            str: Local guideline content or a safe fallback string.
        """

        knowledge_base_dir = self._settings.knowledge_base_dir
        guideline_files = sorted(path for path in knowledge_base_dir.glob("*.md") if path.is_file())
        if guideline_files:
            ranked_files = sorted(
                guideline_files,
                key=lambda path: self._score_fallback_file(path, vitals, route),
                reverse=True,
            )
            selected_files = ranked_files[: self._settings.qdrant_top_k]
            return "\n\n".join(path.read_text(encoding="utf-8") for path in selected_files)

        return (
            "Fallback guideline unavailable. Continue bedside assessment, "
            "support oxygenation, and escalate to the ICU team."
        )

    def _build_fallback_inspection(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
        query: str,
        reason: str,
    ) -> dict[str, object]:
        """Build a debug payload for local fallback retrieval."""

        selected_files = self._rank_fallback_files(vitals, route)[: self._settings.qdrant_top_k]
        chunks: list[RetrievalChunk] = []
        for index, path in enumerate(selected_files, start=1):
            content = path.read_text(encoding="utf-8")
            chunks.append(
                RetrievalChunk(
                    rank=index,
                    score=self._score_fallback_file(path, vitals, route),
                    condition=path.stem.replace("_", " "),
                    source=path.name,
                    chunk_index="fallback-file",
                    preview=content[:280].strip(),
                )
            )

        return {
            "mode": "fallback",
            "route_name": route.name,
            "route_summary": route.summary,
            "query": query,
            "reason": reason,
            "chunks": [chunk.__dict__ for chunk in chunks],
            "guideline_context": "\n\n".join(
                path.read_text(encoding="utf-8")
                for path in selected_files
            )
            if selected_files
            else (
                "Fallback guideline unavailable. Continue bedside assessment, "
                "support oxygenation, and escalate to the ICU team."
            ),
        }

    def _build_chunk_debug_payload(
        self,
        scored_documents: List[tuple[Document, int]],
    ) -> list[dict[str, object]]:
        """Format retrieved chunks for the debug UI."""

        chunks: list[dict[str, object]] = []
        for index, (document, score) in enumerate(scored_documents, start=1):
            metadata = document.metadata
            chunks.append(
                RetrievalChunk(
                    rank=index,
                    score=score,
                    condition=str(metadata.get("condition", "unknown condition")),
                    source=str(metadata.get("file_name", metadata.get("source", "unknown source"))),
                    chunk_index=str(metadata.get("chunk_index", "n/a")),
                    preview=document.page_content.strip()[:280],
                ).__dict__
            )
        return chunks

    def _rank_fallback_files(
        self,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> list[Path]:
        """Return fallback guideline files ranked by lexical relevance."""

        knowledge_base_dir = self._settings.knowledge_base_dir
        guideline_files = sorted(path for path in knowledge_base_dir.glob("*.md") if path.is_file())
        return sorted(
            guideline_files,
            key=lambda path: self._score_fallback_file(path, vitals, route),
            reverse=True,
        )

    def _score_fallback_file(
        self,
        guideline_path: Path,
        vitals: PatientVitals,
        route: RetrievalRoute,
    ) -> int:
        """Score a local guideline file for simple lexical fallback ranking."""

        query_terms = {
            vitals.status.lower(),
            *route.keywords,
            *route.preferred_conditions,
        }
        haystack = f"{guideline_path.stem} {guideline_path.read_text(encoding='utf-8')}".lower()
        return sum(1 for term in query_terms if term and term in haystack)

    def _normalize_recommendations(self, raw_content: object) -> list[str]:
        """Convert an LLM response payload into a stable list of recommendation strings."""

        if isinstance(raw_content, str):
            raw_text = raw_content
        elif isinstance(raw_content, list):
            raw_text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        else:
            raw_text = str(raw_content)

        recommendations: list[str] = []
        seen: set[str] = set()
        for line in raw_text.splitlines():
            cleaned_line = line.lstrip("-*0123456789. )").strip()
            if not cleaned_line:
                continue

            normalized_key = cleaned_line.lower()
            if normalized_key in seen:
                continue

            seen.add(normalized_key)
            recommendations.append(cleaned_line)
            if len(recommendations) >= self._settings.openai_max_recommendations:
                break

        return recommendations

    def _build_vitals_snapshot(self, vitals: PatientVitals) -> str:
        """Build a compact, multi-field vitals snapshot string for retrieval and prompting."""

        parts = [
            f"Heart rate: {vitals.heart_rate:.1f} bpm",
            f"SpO2: {vitals.spo2:.1f} percent",
        ]

        optional_fields = (
            ("Respiratory rate", vitals.respiratory_rate, "/min"),
            ("Systolic BP", vitals.systolic_bp, "mmHg"),
            ("MAP", vitals.mean_arterial_pressure, "mmHg"),
            ("Glucose", vitals.glucose_mg_dl, "mg/dL"),
            ("Lactate", vitals.lactate_mmol_l, "mmol/L"),
            ("Post-op drain output", vitals.postoperative_drain_output_ml_hr, "mL/hr"),
        )
        for label, value, unit in optional_fields:
            if value is not None:
                parts.append(f"{label}: {value:.1f} {unit}")

        return "\n".join(parts)

    def _build_fallback_recommendations(self, vitals: PatientVitals) -> list[str]:
        """Return deterministic bedside-safe guidance when LLM generation is unavailable."""

        return [
            "Confirm pulse oximetry waveform quality and verify sensor placement.",
            "Assess airway, work of breathing, and mental status at bedside.",
            "Increase supplemental oxygen per ICU protocol and reassess promptly.",
            (
                "Escalate to the responsible clinician if SpO2 remains below "
                f"{self._settings.spo2_alert_threshold:.0f}% or the patient deteriorates."
            ),
            "Requires immediate human physician judgment if the bedside picture does not fit the retrieved guidance.",
        ][: self._settings.openai_max_recommendations]
