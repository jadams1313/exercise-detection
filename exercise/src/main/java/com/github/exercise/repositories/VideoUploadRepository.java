package com.github.exercise.repositories;

import com.github.exercise.data.VideoUpload;
import com.github.exercise.constants.VideoStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface VideoUploadRepository extends JpaRepository<VideoUpload, Long> {

    List<VideoUpload> findByUserId(Long userId);

    List<VideoUpload> findByUserIdAndStatus(Long userId, VideoStatus status);

    List<VideoUpload> findByStatus(VideoStatus status);

    long countByUserId(Long userId);
}
