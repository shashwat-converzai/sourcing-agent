from django.core.management.base import BaseCommand
from faker import Faker
import random

from sourcing.models import (
    Candidate, Skill, CandidateSkill, Experience,
    Job, JobSkill
)


class Command(BaseCommand):
    help = "Populate dummy data for POC"

    def handle(self, *args, **options):
        fake = Faker()

        # -----------------------------
        # 1. Seed Skills
        # -----------------------------
        skill_names = [
            "Java", "Python", "Django", "React", "Node.js",
            "AWS", "Azure", "SQL", "NoSQL",
            "HR Management", "Recruitment", "Excel",
            "Accounting", "Bookkeeping", "Financial Analysis",
            "Welding", "Mechanical Repair", "Plumbing"
        ]

        skills = []
        for name in skill_names:
            s, _ = Skill.objects.get_or_create(name=name)
            skills.append(s)

        # -----------------------------
        # 2. Seed Candidates
        # -----------------------------
        industries = ["IT", "Logistics", "Healthcare"]
        domains = [
            "software", "hr", "finance",
            "accounts", "business", "admin",
            "doctor", "nurse", "mechanic"
        ]
        seniority_levels = ["junior", "mid-level", "mid-senior", "senior"]

        self.stdout.write("Creating candidates...")
        candidates = []

        for _ in range(50):
            name = fake.name()
            candidate = Candidate.objects.create(
                candidate_id=f"C{random.randint(1000, 9999)}",
                full_name=name,
                email=fake.email(),
                phone=fake.phone_number(),
                present_location=fake.city(),
                permanent_location=fake.city(),
                total_experience_years=random.randint(1, 15),
                current_designation=random.choice([
                    "Software Engineer", "HR Executive", "Accountant", "Mechanic"
                ]),
                industry=random.choice(industries),
                domain=random.choice(domains),
                seniority_level=random.choice(seniority_levels),
                resume_raw=fake.text(500),
            )

            # Assign random skills
            picked_skills = random.sample(skills, random.randint(3, 6))
            for sk in picked_skills:
                CandidateSkill.objects.create(
                    candidate=candidate,
                    skill=sk,
                    proficiency=random.choice(
                        ["beginner", "intermediate", "expert"]),
                    years_of_experience=random.randint(1, 10),
                    last_used_year=random.randint(2015, 2024),
                )

            # Add experience entries
            for _ in range(random.randint(1, 3)):
                Experience.objects.create(
                    candidate=candidate,
                    company_name=fake.company(),
                    designation=candidate.current_designation,
                    start_date=fake.date_between(
                        start_date="-10y", end_date="-1y"),
                    end_date=fake.date_between(
                        start_date="-1y", end_date="today"),
                    responsibilities=fake.text(120),
                )

            candidates.append(candidate)

        # -----------------------------
        # 3. Seed Jobs
        # -----------------------------
        self.stdout.write("Creating jobs...")

        job_titles = [
            "Backend Developer", "Frontend Developer", "Data Engineer",
            "HR Manager", "Accountant", "Mechanical Technician"
        ]

        jobs = []
        for i in range(20):
            title = random.choice(job_titles)
            job = Job.objects.create(
                job_id=f"J{i+1}",
                title=title,
                role_code=f"R{i+1}",
                description=fake.text(300),
                location=fake.city(),
                experience_min_years=random.randint(1, 3),
                experience_max_years=random.randint(5, 10),
                industry=random.choice(industries),
                domain=random.choice(domains),
                seniority_level=random.choice(seniority_levels),
            )

            # Assign job skills
            picked_skills = random.sample(skills, random.randint(3, 6))
            for sk in picked_skills:
                JobSkill.objects.create(
                    job=job,
                    skill=sk,
                    importance=random.choice(["must", "nice"]),
                )

            jobs.append(job)

        self.stdout.write(self.style.SUCCESS(
            "Dummy data created successfully!"))
