package com.github.exercise.controllers;

import com.github.exercise.data.User;
import com.github.exercise.dto.VideoUploadResponse;
import com.github.exercise.data.VideoUpload;
import com.github.exercise.service.FileStorageService;
import com.github.exercise.service.UserService;
import com.github.exercise.service.VideoUploadService;
import lombok.RequiredArgsConstructor;
import org.apache.coyote.Response;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/videos")
@RequiredArgsConstructor
public class VideoUploadController {

    private final VideoUploadService videoUploadService;
    private final FileStorageService fileStorageService;
    private final UserService userService;

    @PostMapping("/upload")
    public ResponseEntity<VideoUploadResponse> uploadVideo(
            @RequestParam("file") MultipartFile file,
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());
        User user = getUserOrThrow(userId);

        try {
            VideoUpload video = videoUploadService.uploadVideo(user, file);
            VideoUploadResponse response = mapToResponse(video);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (IOException e) {
            throw new RuntimeException("Failed to upload video", e);
        }
    }

    @GetMapping("/{videoId}")
    public ResponseEntity<VideoUploadResponse> getVideo(
            @PathVariable Long videoId,
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());
        VideoUpload video = videoUploadService.getVideoById(videoId, userId);

        VideoUploadResponse response = mapToResponse(video);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<List<VideoUploadResponse>> getAllUserVideos(
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());
        User user = getUserOrThrow(userId);
        List<VideoUpload> videos = videoUploadService.getAllUserVideos(user);

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

        Long userId = Long.parseLong(authentication.getName());
        VideoUpload video = videoUploadService.getVideoById(videoId, userId);

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

        Long userId = Long.parseLong(authentication.getName());
        videoUploadService.deleteVideo(videoId, userId);

        return ResponseEntity.noContent().build();
    }

    private VideoUploadResponse mapToResponse(VideoUpload videoUpload) {
        VideoUploadResponse response = new VideoUploadResponse();
        response.setId(videoUpload.getId());
        response.setOriginalFileName(videoUpload.getFilename());
        //response.setS3Key(videoUpload.getS3Key());
        response.setFileSize(videoUpload.getFileSizeBytes());
        response.setStatus(videoUpload.getStatus());
        response.setUploadedAt(videoUpload.getUploadedAt());
        return response;
    }
    private User getUserOrThrow(Long userId) {
        return userService.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));
    }
}