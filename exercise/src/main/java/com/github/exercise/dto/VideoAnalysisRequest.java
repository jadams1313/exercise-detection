package com.github.exercise.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class VideoAnalysisRequest {
    private String videoData; // base64 encoded video
    private String fileName;
}