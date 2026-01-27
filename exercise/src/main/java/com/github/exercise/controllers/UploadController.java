package com.github.exercise.controllers;

import com.github.exercise.dto.VideoUploadResponse;
import com.github.exercise.data.VideoUpload;
import com.github.exercise.service.FileStorageService;
import com.github.exercise.service.VideoUploadService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/videos")
@RequiredArgsConstructor
public class VideoUploadController {

    private final VideoUploadService videoUploadService;
    private final FileStorageService fileStorageService;

    @PostMapping("/upload")
    public ResponseEntity<VideoUploadResponse> uploadVideo(
            @RequestParam("file") MultipartFile file,
            Authentication authentication) {

        String username = authentication.getName();
        VideoUpload video = videoUploadService.uploadVideo(file, username);

        VideoUploadResponse response = mapToResponse(video);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/{videoId}")
    public ResponseEntity<VideoUploadResponse> getVideo(
            @PathVariable Long videoId,
            Authentication authentication) {

        String username = authentication.getName();
        VideoUpload video = videoUploadService.getVideoById(videoId, username);

        VideoUploadResponse response = mapToResponse(video);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<List<VideoUploadResponse>> getAllUserVideos(Authentication authentication) {
        String username = authentication.getName();
        List<VideoUpload> videos = videoUploadService.getAllUserVideos(username);

        List<VideoUploadResponse> responses = videos.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());

        return ResponseEntity.ok(responses);
    }

    @GetMapping("/{videoId}/url")
    public ResponseEntity<String> getVideoUrl(
            @PathVariable Long videoId,
            @RequestParam(defaultValue = "60") int expirationMinutes,
            Authentication authentication) {

        String username = authentication.getName();
        VideoUpload video = videoUploadService.getVideoById(videoId, username);

        String presignedUrl = fileStorageService.getPresignedUrl(
                video.getFilename(),
                expirationMinutes
        );

        return ResponseEntity.ok(presignedUrl);
    }

    @DeleteMapping("/{videoId}")
    public ResponseEntity<Void> deleteVideo(
            @PathVariable Long videoId,
            Authentication authentication) {

        String username = authentication.getName();
        videoUploadService.deleteVideo(videoId, username);

        return ResponseEntity.noContent().build();
    }

    private VideoUploadResponse mapToResponse(VideoUpload video) {
        return VideoUploadResponse.builder()
                .id(video.getId())
                .originalFileName(video.getFilename())
                .status(video.getStatus())
                .uploadedAt(video.getUploadedAt())
                .fileSize(video.getFileSizeBytes())
                .build();
    }
}