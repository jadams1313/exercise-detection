package com.github.exercise.controllers;

import com.github.exercise.dto.AnalysisResponse;
import com.github.exercise.data.ExerciseAnalysis;
import com.github.exercise.service.ExerciseAnalysisService;
import com.github.exercise.service.VideoUploadService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/analysis")
@RequiredArgsConstructor
public class ExerciseAnalysisController {

    private final ExerciseAnalysisService analysisService;
    private final VideoUploadService videoUploadService;

    @GetMapping("/video/{videoId}")
    public ResponseEntity<AnalysisResponse> getAnalysisByVideoId(
            @PathVariable Long videoId,
            Authentication authentication) {

        String username = authentication.getName();
        // Verify user owns the video
        videoUploadService.getVideoById(videoId, username);

        ExerciseAnalysis analysis = analysisService.getAnalysisByVideoId(videoId);
        AnalysisResponse response = mapToResponse(analysis);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/{analysisId}")
    public ResponseEntity<AnalysisResponse> getAnalysisById(
            @PathVariable Long analysisId,
            Authentication authentication) {

        ExerciseAnalysis analysis = analysisService.getAnalysisById(analysisId);

        // Verify user owns the video
        String username = authentication.getName();
        videoUploadService.getVideoById(analysis.getVideoUpload().getId(), username);

        AnalysisResponse response = mapToResponse(analysis);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/user")
    public ResponseEntity<List<AnalysisResponse>> getUserAnalyses(Authentication authentication) {
        String username = authentication.getName();

        // Get user's analyses through service (we need to add this method)
        List<ExerciseAnalysis> analyses = analysisService.getAnalysesByUsername(username);

        List<AnalysisResponse> responses = analyses.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());

        return ResponseEntity.ok(responses);
    }

    private AnalysisResponse mapToResponse(ExerciseAnalysis analysis) {
        return AnalysisResponse.builder()
                .id(analysis.getId())
                .videoId(analysis.getVideoUpload().getId())
                .exerciseType(analysis.getExerciseType())
                .repCount(analysis.getRepCount())
                .confidence(analysis.getConfidence())
                .status(analysis.getStatus())
                .createdAt(analysis.getCreatedAt())
                .completedAt(analysis.getCompletedAt())
                .errorMessage(analysis.getErrorMessage())
                .build();
    }
}