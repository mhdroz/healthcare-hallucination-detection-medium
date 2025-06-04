// Healthcare RAG System Frontend JavaScript - FastAPI Version

class HealthcareRAGApp {
    constructor() {
        this.apiBase = '/api';
        this.initializeEventListeners();
        this.checkSystemHealth();
        this.loadSampleQuestions();
    }

    initializeEventListeners() {
        // Query form submission
        document.getElementById('queryForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleQuery();
        });

        // Evaluation button
        document.getElementById('evaluateBtn').addEventListener('click', () => {
            this.handleEvaluation();
        });

        // Sample question buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('sample-question-btn')) {
                const question = e.target.dataset.question;
                document.getElementById('question').value = question;
            }
        });

        // Initialize tooltips
        this.initializeTooltips();
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy' && data.models_loaded) {
                console.log('✅ System healthy and ready');
                this.showHealthStatus(true);
            } else {
                this.showError('System not ready. Please wait for models to load.');
                this.showHealthStatus(false);
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.showError('Unable to connect to the system.');
            this.showHealthStatus(false);
        }
    }

    async loadSampleQuestions() {
        try {
            const response = await fetch(`${this.apiBase}/sample-questions`);
            const data = await response.json();
            this.displaySampleQuestions(data.samples);
        } catch (error) {
            console.error('Failed to load sample questions:', error);
        }
    }

    displaySampleQuestions(questions) {
        const container = document.getElementById('sampleQuestions');
        if (container) {
            const html = questions.map((q, i) => 
                `<button type="button" class="btn btn-outline-secondary btn-sm sample-question-btn mb-1" 
                         data-question="${this.escapeHtml(q)}" title="Click to use this question">
                    ${this.escapeHtml(q.substring(0, 50))}...
                </button>`
            ).join('');
            container.innerHTML = html;
        }
    }

    showHealthStatus(healthy) {
        const indicator = document.getElementById('healthIndicator');
        if (indicator) {
            indicator.className = healthy ? 'badge bg-success' : 'badge bg-danger';
            indicator.textContent = healthy ? 'System Ready' : 'System Not Ready';
        }
    }

    async handleQuery() {
        const question = document.getElementById('question').value.trim();
        
        if (!question) {
            this.showError('Please enter a question.');
            return;
        }

        // Collect configuration
        const config = {
            question: question,
            fact_check: document.getElementById('factCheck').checked,
            multi_stage: document.getElementById('multiStage').checked,
            consistency_tries: parseInt(document.getElementById('consistencyTries').value)
        };

        this.showLoading();

        try {
            const response = await fetch(`${this.apiBase}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Query failed');
            }

            const data = await response.json();
            this.displayResults(data);

        } catch (error) {
            console.error('Query error:', error);
            this.showError(error.message);
        }
    }

    async handleEvaluation() {
        const button = document.getElementById('evaluateBtn');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Running Evaluation...';
        button.disabled = true;

        try {
            const response = await fetch(`${this.apiBase}/evaluate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Evaluation failed');
            }

            const data = await response.json();
            this.displayEvaluationResults(data);

        } catch (error) {
            console.error('Evaluation error:', error);
            this.showError('Evaluation failed: ' + error.message);
        } finally {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    showLoading() {
        this.hideAllStates();
        document.getElementById('loadingState').style.display = 'block';
    }

    showError(message) {
        this.hideAllStates();
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorState').style.display = 'block';
    }

    hideAllStates() {
        document.getElementById('welcomeState').style.display = 'none';
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('resultsState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';
    }

    displayResults(data) {
        this.hideAllStates();
        
        // Update question and answer
        document.getElementById('questionDisplay').innerHTML = 
            `<strong>Question:</strong> ${this.escapeHtml(data.question)}`;
        document.getElementById('answerDisplay').innerHTML = this.escapeHtml(data.answer);

        // Update safety score
        this.updateSafetyScore(data);

        // Update individual metrics
        this.updateSafetyMetrics(data);

        // Update weak sentences
        this.updateWeakSentences(data);

        // Show results with animation
        document.getElementById('resultsState').style.display = 'block';
        document.getElementById('resultsState').classList.add('slide-in');
    }

    updateSafetyScore(data) {
        const scoreRatio = data.safety_score / data.max_safety_score;
        const scoreText = `${data.safety_score}/${data.max_safety_score}`;
        
        // Update score display
        document.getElementById('safetyScoreText').textContent = scoreText;
        
        // Update confidence badge and circle
        const badge = document.getElementById('confidenceBadge');
        const circle = document.getElementById('safetyScoreCircle');
        
        // Remove existing classes
        circle.classList.remove('high-confidence', 'medium-confidence', 'low-confidence');
        badge.classList.remove('bg-success', 'bg-warning', 'bg-danger', 'high-confidence', 'medium-confidence', 'low-confidence');
        
        if (data.confidence === 'HIGH CONFIDENCE') {
            circle.classList.add('high-confidence');
            badge.classList.add('bg-success', 'high-confidence');
        } else if (data.confidence === 'MEDIUM CONFIDENCE') {
            circle.classList.add('medium-confidence');
            badge.classList.add('bg-warning', 'medium-confidence');
        } else {
            circle.classList.add('low-confidence');
            badge.classList.add('bg-danger', 'low-confidence');
        }
        
        badge.textContent = data.confidence;
    }

    updateSafetyMetrics(data) {
        // Attribution
        this.updateMetricBar('attribution', data.attribution_score, 
            data.safety_interpretations.attribution);
        
        // Consistency
        this.updateMetricBar('consistency', data.consistency_score,
            data.safety_interpretations.consistency);
        
        // Entropy (invert for display - lower entropy is better)
        const entropyDisplay = Math.max(0, 1 - (data.semantic_entropy / 3));
        this.updateMetricBar('entropy', entropyDisplay,
            data.safety_interpretations.entropy);
        
        // External validation
        if (data.external_validation && !data.external_validation.error) {
            this.updateMetricBar('external', data.external_validation.score,
                data.external_validation.interpretation ? 
                data.external_validation.interpretation.interpretation : 
                'External validation completed');
        } else if (data.external_validation && data.external_validation.error) {
            this.updateMetricBar('external', 0, 
                `External validation failed: ${data.external_validation.error}`);
        } else {
            this.updateMetricBar('external', 0, 'External validation disabled');
        }
    }

    updateMetricBar(metric, score, interpretation) {
        const progress = document.getElementById(`${metric}Progress`);
        const text = document.getElementById(`${metric}Text`);
        
        // Update progress bar
        const percentage = Math.round(score * 100);
        progress.style.width = `${percentage}%`;
        progress.setAttribute('aria-valuenow', percentage);
        
        // Update text
        text.textContent = `${interpretation} (${(score * 100).toFixed(1)}%)`;
        
        // Color coding
        progress.classList.remove('bg-success', 'bg-warning', 'bg-danger');
        if (score >= 0.7) {
            progress.classList.add('bg-success');
        } else if (score >= 0.5) {
            progress.classList.add('bg-warning');
        } else {
            progress.classList.add('bg-danger');
        }
    }

    updateWeakSentences(data) {
        const container = document.getElementById('weakSentences');
        const list = document.getElementById('weakSentencesList');
        
        if (data.has_weak_sentences) {
            container.style.display = 'block';
            list.innerHTML = '';
            
            data.weak_sentences.forEach(weak => {
                const div = document.createElement('div');
                div.className = 'weak-sentence';
                div.innerHTML = `
                    <strong>Score: ${(weak.score * 100).toFixed(1)}%</strong><br>
                    ${this.escapeHtml(weak.sentence)}
                `;
                list.appendChild(div);
            });
        } else {
            container.style.display = 'none';
        }
    }

    displayEvaluationResults(data) {
        const container = document.getElementById('evaluationResults');
        
        const html = `
            <h6>RAGAS Evaluation Results</h6>
            <div class="evaluation-metric">
                <span>Faithfulness</span>
                <span class="metric-score ${this.getScoreClass(data.faithfulness_score)}">
                    ${(data.faithfulness_score * 100).toFixed(1)}%
                </span>
            </div>
            <div class="evaluation-metric">
                <span>Answer Relevancy</span>
                <span class="metric-score ${this.getScoreClass(data.relevancy_score)}">
                    ${(data.relevancy_score * 100).toFixed(1)}%
                </span>
            </div>
            <div class="evaluation-metric">
                <span>Overall Grade</span>
                <span class="metric-score ${this.getGradeClass(data.overall_grade)}">
                    ${data.overall_grade}
                </span>
            </div>
            <small class="text-muted">
                Evaluated on ${data.num_questions} pneumonia-related questions
            </small>
        `;
        
        container.innerHTML = html;
        container.style.display = 'block';
        container.classList.add('fade-in');
    }

    getScoreClass(score) {
        if (score >= 0.8) return 'excellent';
        if (score >= 0.6) return 'good';
        if (score >= 0.4) return 'fair';
        return 'poor';
    }

    getGradeClass(grade) {
        if (grade.startsWith('A')) return 'excellent';
        if (grade.startsWith('B')) return 'good';
        if (grade.startsWith('C')) return 'fair';
        return 'poor';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HealthcareRAGApp();
});