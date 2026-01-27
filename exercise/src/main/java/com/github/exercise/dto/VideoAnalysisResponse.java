package com.github.exercise.dto;

public class VideoAnalysisResponse {
    private String exerciseType;
    private Integer repCount;
    private Double confidence;
    
    public String getExerciseType() {
        return this.exerciseType; 
    }
    public Integer getRepCount() {
        return this.repCount;
    }
    public Double getConfidence() {
        return this.confidence;
    }
    public void setExerciseType(final String exerciseType) {
        this.exerciseType = exerciseType;
    }
    public void setRepCount(final Integer repCount) {
        this.repCount = repCount;
    }
    public void setConfidence(final Double confidence) {
        this.confidence = confidence;

    }
}