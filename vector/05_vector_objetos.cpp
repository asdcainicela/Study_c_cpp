#include <iostream>
#include <vector>
#include <string>

struct Person {
    std::string name;
    int age;
    Person(std::string n, int a) : name(std::move(n)), age(a) {}
};

int main() {
    std::vector<Person> people;
    people.emplace_back("Alice", 30);
    people.emplace_back("Bob", 25);

    for (const auto& p : people)
        std::cout << p.name << " (" << p.age << ")\n";

    std::vector<Person*> ptrs;
    ptrs.push_back(&people[0]);
    ptrs.push_back(&people[1]);

    for (auto* p : ptrs)
        std::cout << "Ptr -> " << p->name << std::endl;

    return 0;
}
