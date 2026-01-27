package com.github.exercise.dto;
import com.github.exercise.constants.AnalysisStatus;
import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class AnalysisResponse {
    private Long id;
    private Long videoId;
    private String exerciseType;
    private Integer repCount;
    private Double confidence;
    private AnalysisStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime completedAt;
    private String errorMessage;
}