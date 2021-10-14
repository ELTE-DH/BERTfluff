from random import randrange, shuffle

from sqlalchemy.orm import declarative_base
from sqlalchemy import func, Table, create_engine
from sqlalchemy.orm import Session


class ContextBank:
    def __init__(self, db_config: dict, db=None, window_size: int = 11, hide_char: str = '#'):
        """
        Interface for selecting words and appropriate contexts for them

        :param db_config: A dictionary containing the database configuration. Mandatory keys:  database_name,
            table_name, id_name, left_name, word_name, right_name, freq_name
        :param db: An initialized flask_sqlalchemy.SQLAlchemy object
        :param window_size: the window's size left context + word + right context
        :param hide_char: Character to use when hiding word
        """

        if db is None and 'database_name' in db_config:
            self._db = create_engine(f'sqlite:///{db_config["database_name"]}')
            self._session = Session(self._db)
        elif db is not None and hasattr(db, 'session'):
            self._db = db
            self._session = db.session
        else:
            raise ValueError('db_config[\'database_name\'] or db from flask_sqlalchemy.SQLAlchemy must be set!')

        self._table_obj = Table(db_config['table_name'], declarative_base().metadata, autoload_with=self._db.engine)
        col_objs = {col_obj.key: col_obj for col_obj in self._table_obj.c}
        self._id_obj = col_objs[db_config['id_name']]
        self._left_obj = col_objs[db_config['left_name']]
        self._word_obj = col_objs[db_config['word_name']]
        self._right_obj = col_objs[db_config['right_name']]
        self._freq = col_objs[db_config['freq_name']]

        self._window_size = window_size
        self._hide_char = hide_char

    def read_all_lines_for_word(self, word: str = None, displayed_lines: list = (), hide_word=True):
        """Read all lines for the specific word and separate the ones which were already shown from the new ones"""

        if self._window_size > 2:
            context_size = (self._window_size - 1) // 2
        else:
            context_size = 1_000_000  # Extremely big to include full sentence

        if hide_word:
            hide_fun = self._hide_word
        else:
            hide_fun = self._identity

        lines_to_display, new_lines = {}, []
        if word is None:
            if len(displayed_lines) > 0:
                word, _ = self.identify_word_from_id(displayed_lines[0])
            else:
                ValueError('At least one displayed line must present to look up the word'
                           ' if word is None in read_all_lines_for_word !')
        displayed_lines_set = set(displayed_lines)

        all_lines_for_word_query = self._session.query(self._table_obj). \
            with_entities(self._id_obj, self._left_obj, self._word_obj, self._right_obj). \
            filter(self._word_obj == word)
        for line_id, left, word, right in all_lines_for_word_query.all():
            word = hide_fun(word)
            # Truncate contexts if needed
            left_truncated = ' '.join(left.split(' ')[-min(context_size, len(left)):])
            right_truncated = ' '.join(right.split(' ')[:min(context_size, len(right))])

            if line_id in displayed_lines_set:
                lines_to_display[line_id] = [line_id, left_truncated, word, right_truncated]
            else:
                new_lines.append([line_id, left_truncated, word, right_truncated])

        lines_to_display = [lines_to_display[line_id] for line_id in displayed_lines]  # In the original order!
        shuffle(new_lines)

        return lines_to_display, new_lines

    def select_one_random_line(self):
        """Select one random line from all available lines
            Raises sqlalchemy.orm.exc.NoResultFound if the query selects no rows
            Raises sqlalchemy.orm.exc.MultipleResultsFound if multiple rows are returned
        """

        random_line_id = self._get_random_line_id()

        # Retrieve data for that specific line
        entry_query = self._session.query(self._table_obj). \
            with_entities(self._id_obj, self._left_obj, self._word_obj, self._right_obj). \
            filter(self._id_obj == random_line_id)
        line_id, left, word, right = entry_query.one()

        return [[line_id, left, self._hide_word(word), right]]

    def _get_random_line_id(self):
        """Select a random id (line_id) from the table"""
        row_count_query = self._session.query(func.count(self._id_obj))
        row_count = row_count_query.scalar()
        random_line_id = randrange(row_count) + 1

        return random_line_id

    def select_random_word(self):
        """Select one random word from all available lines
            Raises sqlalchemy.orm.exc.NoResultFound if the query selects no rows
            Raises sqlalchemy.orm.exc.MultipleResultsFound if multiple rows are returned
        """

        random_line_id = self._get_random_line_id()

        return self.identify_word_from_id(random_line_id)

    def identify_word_from_id(self, one_line_id):
        """Identify word from (one of the the already shown) line ID
            Raises sqlalchemy.orm.exc.NoResultFound if the query selects no rows
            Raises sqlalchemy.orm.exc.MultipleResultsFound if multiple rows are returned
        """

        word_query = self._session.query(self._table_obj). \
            with_entities(self._word_obj, self._freq). \
            filter(self._id_obj == one_line_id)
        word, freq = word_query.one()

        return word, freq

    @staticmethod
    def _identity(word):
        return word

    def _hide_word(self, word: str):
        """Hide word with required amount of self._hide_char characters to maintain the length"""
        return self._hide_char * len(word)
