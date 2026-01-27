package com.github.exercise.dto;
import com.github.exercise.constants.AnalysisStatus;
import com.github.exercise.constants.VideoStatus;
import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class VideoUploadResponse {
    private Long id;
    private String originalFileName;
    private VideoStatus status;
    private LocalDateTime uploadedAt;
    private Long fileSize;
}