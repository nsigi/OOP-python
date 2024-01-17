import datetime
from typing import Union, List, Dict
from collections import namedtuple
from datetime import date
import os.path
import json
import pandas as pd


LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")
STUDENT_ARGS_CAST = {"unique_id": int, "name": str, "surname": str, "group": int, "subgroup": int}


def bool_to_int(val):
    return 1 if val else 0


class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_work_number, lab_work_mark, lab_work_date')):
    """
    Информация о лабораторном занятии, которое могло или не могло быть посещено студентом
    """

    def __new__(cls, presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date):
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if LabWorkSession._validate_session(presence, lab_work_number, lab_work_mark, lab_work_date):
            return super().__new__(cls, presence, lab_work_number, lab_work_mark, lab_work_date)

        raise ValueError(f"LabWorkSession ::"
                         f"incorrect args :\n"
                         f"presence       : {presence},\n"
                         f"lab_work_number: {lab_work_number},\n"
                         f"lab_work_mark  : {lab_work_mark},\n"
                         f"lab_work_date  : {lab_work_date}")

    @staticmethod
    def _validate_session(presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date) -> bool:
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if not presence:
            return  False
        if not isinstance(lab_work_number, int) and lab_work_number < 1:
            return False
        if not isinstance(lab_work_mark, int) and not lab_work_mark in {2, 3, 4, 5}:
            return False
        if not isinstance(lab_work_date, date):
            return False
        return True

    def __str__(self) -> str:
        date = datetime.datetime.strftime(self.lab_work_date, '%d:%m:%y')
        #date = datetime.datetime.timepstr(self.lab_work_date, '%d:%m:%y').date()
        return "\n\t{\n" \
               f"\t\t\"presence\": {bool_to_int(self.presence)},\n" \
               f"\t\t\"lab_work_n\": {self.lab_work_number},\n" \
               f"\t\t\"lab_work_mark\"  : {self.lab_work_mark},\n" \
               f"\t\t\"date\"  : \"{date}\"\n" \
               "\t}"


class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):

        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not Student._validate_args_student(unique_id, name, surname, group, subgroup):
            raise ValueError(f"Student ::"
                             f"incorrect args :\n"
                             f"unique_id: {unique_id},\n"
                             f"name: {name},\n"
                             f"surname  : {surname},\n"
                             f"group  : {group}, \n"
                             f"subgroup  : {subgroup}")

        self._unique_id = unique_id
        self._name = name
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        self._lab_work_sessions = []

    @staticmethod
    def _validate_args_student(unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        #type(unique_id)
        if not isinstance(unique_id, int):
            return False
        if not isinstance(name, str):
            return False
        if len(name) == 0:
            return False
        if not isinstance(surname, str):
            return False
        if len(surname) == 0:
            return False
        if not isinstance(group, int):
            return False
        if not isinstance(subgroup, int):
            return False

        return True

    def __str__(self) -> str:
        sep = ',\n'
        return "\t{\n" \
               f"\t\"unique_id\": {self._unique_id},\n" \
               f"\t\"name\": \"{self.name}\",\n" \
               f"\t\"surname\": \"{self._surname}\",\n" \
               f"\t\"group\": {self.group},\n" \
               f"\t\"subgroup\": {self.subgroup},\n"\
               f"\t\"lab_works_sessions\": [{sep.join(str(v)for v in self.lab_work_sessions)}]\n"\
               "\t}"

    @property
    def unique_id(self) -> int:
        """
        Метод доступа для unique_id
        """
        return self._unique_id

    @property
    def group(self) -> int:
        """
        Метод доступа для номера группы
        """
        return self._group

    @property
    def subgroup(self) -> int:
        """
        Метод доступа для номера подгруппы
        """
        return self._subgroup

    @property
    def name(self) -> str:
        """
        Метод доступа для имени студента
        """
        return self._name

    @property
    def surname(self) -> str:
        """
        Метод доступа для фамилии студента
        """
        return self._surname

    @name.setter
    def name(self, val: str) -> None:
        """
        Метод для изменения значения имени студента
        """
        self._name = val

    @surname.setter
    def surname(self, val: str) -> None:
        """
        Метод для изменения значения фамилии студента
        """
        self._surname = val

    @property
    def lab_work_sessions(self):
        """
        Метод доступа для списка лабораторных работ, которые студент посетил или не посетил
        """
        for item in self._lab_work_sessions:
            yield item

    def append_lab_work_session(self, session: LabWorkSession):
        """
        Метод для регистрации нового лабораторного занятия
        """
        self._lab_work_sessions.append(session)


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(f"load_lab_work_session:: key \"{key}\" not present in json_node")
    return create_session(json_node['precence'], json_node['lab_work_n'], json_node['lab_work_mark'], json_node['date'])


def create_student(params):
    return Student(int(params[UNIQUE_ID]), params[STUD_NAME], params[STUD_SURNAME], int(params[STUD_GROUP]), int(params[STUD_SUBGROUP]))


def create_session(presence, number, mark, date):
    return LabWorkSession(True if presence == 1 else False, int(number), int(mark), datetime.datetime.strptime(date, '%d:%m:%y').date())


def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    dictStudent = {}
    for key in STUDENT_KEYS:
        dictStudent[key] = json_node[key]
    '''student = create_student([dictStudent[STUDENT_KEYS[UNIQUE_ID]], dictStudent[STUDENT_KEYS[STUD_NAME]],
                      dictStudent[STUDENT_KEYS[STUD_SURNAME]], dictStudent[STUDENT_KEYS[STUD_GROUP]],
                      dictStudent[STUDENT_KEYS[STUD_SUBGROUP]]])'''
    student = Student(*tuple(func(dictStudent[key]) for key, func in STUDENT_ARGS_CAST.items()))
    for session in dictStudent['lab_works_sessions']:
        try:
            student.append_lab_work_session(create_session(session['presence'], session['lab_work_n'],
                                                           session['lab_work_mark'], session['date']))
        except Exception as e:
            print(e)
            continue
    return student

# csv header
#     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
# unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
UNIQUE_ID = 0
STUD_NAME = 1
STUD_SURNAME = 2
STUD_GROUP = 3
STUD_SUBGROUP = 4
LAB_WORK_DATE = 5
STUD_PRESENCE = 6
LAB_WORK_NUMBER = 7
LAB_WORK_MARK = 8


def load_students_csv(file_path: str) -> Union[List[Student], None]:
    # csv header
    #     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
    # unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rt', encoding='utf-8') as input_file:
        data = input_file.readline()
        students = []
        setIDs = set()
        while True:
            #data = pd.read_csv(file_path, sep=';')
            #labWorkSessions = {data['unique_id']: set() for i in range(len(data['unique_id']))}
            #for i in range(len(data)):
            try:
                data = input_file.readline().split(';')
                if (len(data) == 1 and data[0] == ''):
                    break
                for i in range(len(data)):
                    data[i] = data[i].replace("\"", "")
                for i in [0, 3, 4, 6, 7, 8]:
                    data[i] = int(data[i])

                if data[UNIQUE_ID] not in setIDs:
                    students.append(create_student([data[UNIQUE_ID], data[STUD_NAME], data[STUD_SURNAME], data[STUD_GROUP],
                                             data[STUD_SUBGROUP]]))
                    setIDs.add(data[UNIQUE_ID])
                    #student = Student(data['unique_id'][i], data['name'][i], data['surname'][i], data['group'][i], data['subgroup'][i])
                students[data[UNIQUE_ID]].append_lab_work_session(create_session(data[STUD_PRESENCE], data[LAB_WORK_NUMBER],
                                                                                 data[LAB_WORK_MARK], data[LAB_WORK_DATE]))
            except Exception as e:
                print(e)
                continue
        return students


def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rt', encoding='utf-8') as input_file:
        data = json.load(input_file)
        students = []
        for node in data["students"]:
            try:
                students.append(_load_student(node))
            except Exception as e:
                print(e)
                continue
        return students


def save_students_json(file_path: str, students: List[Student]):
    """
    Запись списка студентов в json файл
    """
    with open(file_path, 'wt', encoding="utf-8") as output_file:
        sep = ",\n"
        print(f"{{\n\"students\":[\n{sep.join(str(student) for student in students)}]\n}}", file=output_file)


def to_str_csv(obj):
    sep = ";"
    return sep.join(f"\"{el}\"" if isinstance(el, str) else str(el) for el in obj)


def save_students_csv(file_path: str, students: List[Student]):
    """
    Запись списка студентов в csv файл
    """
    #with open(file_path, 'wt', encoding="utf-8") as output_file:
    file = open(file_path, "w")
    file.write('unique_id;name;surname;group;subgroup;date;presence;lab_work_number;lab_work_mark\n')
    for student in students:
        sep = ";"
        studentStr = to_str_csv([student.unique_id, student.name, student.surname, student.group, student.subgroup])
        for session in student._lab_work_sessions:
            date = datetime.datetime.strftime(session.lab_work_date, '%d:%m:%y')
            presence = bool_to_int(session.presence)
            file.write(studentStr + ";" + to_str_csv([date, presence, session.lab_work_number, session.lab_work_mark])+"\n")
        #sep = ";"
       # print(f"{{\n\t\"students\":[\n{sep.join(str(student) for student in students)}]\n}}", file=output_file)

if __name__ == '__main__':
    # Задание на проверку json читалки:
    # 1. прочитать файл "students.json"
    # 2. сохранить прочитанный файл в "saved_students.json"
    # 3. прочитать файл "saved_students.json"
    # Задание на проверку csv читалки:
    # 1.-3. аналогично

    '''students = load_students_json('students.json')
    #print(*students)
    save_students_json('students_saved.json', students)
    students = load_students_json('students_saved.json')
    print('\n'.join(str(v)for v in students))'''


    students = load_students_csv('students.csv')
    #print(*students)
    save_students_csv('students_saved.csv', students)
    students = load_students_csv('students_saved.csv')
    print(*students)

    '''x = Student(0, 'Nick', 'Samon', 3, 1)
    x.append_lab_work_session(LabWorkSession(True, 1, 4, datetime.date(2020, 7, 14)))
    x.append_lab_work_session(LabWorkSession(True, 2, 5, datetime.date(2020, 7, 15)))
    #x.append_lab_work_session(LabWorkSession(False, 2, 5, datetime.date(2020, 7, 15)))
    print(x)'''

