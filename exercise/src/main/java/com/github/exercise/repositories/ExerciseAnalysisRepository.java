package com.github.exercise.repositories;
import com.github.exercise.constants.AnalysisStatus;
import com.github.exercise.data.ExerciseAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ExerciseAnalysisRepository extends JpaRepository<ExerciseAnalysis, Long> {

    Optional<ExerciseAnalysis> findByVideoUploadId(Long videoUploadId);

    List<ExerciseAnalysis> findByVideoUploadUserId(Long userId);

    List<ExerciseAnalysis> findByStatus(AnalysisStatus status);

    List<ExerciseAnalysis> findByVideoUploadUserIdAndStatus(Long userId, AnalysisStatus status);

    boolean existsByVideoUploadId(Long videoUploadId);


}