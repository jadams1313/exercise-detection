package com.github.exercise.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class VideoAnalysisRequest {
    private String videoData; // base64 encoded video
    private String fileName;

    public String getVideoData() {
        return videoData;
    }

    public void setVideoData(String videoData) {
        this.videoData = videoData;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }
}